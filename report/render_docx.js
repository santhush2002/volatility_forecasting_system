// Generic renderer: report_content.json -> final_report.docx (docx-js)
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType, Table, TableRow, TableCell,
  WidthType, ShadingType, ImageRun, PageBreak, LevelFormat, Footer, PageNumber, TableOfContents,
  BorderStyle,
} = require("docx");

const src = process.argv[2], out = process.argv[3];
const C = JSON.parse(fs.readFileSync(src, "utf8"));
const ROOT = path.dirname(src);
const FONT = "Calibri";
const PAGE_W = 11906, MARGIN = 1134, CONTENT_W = PAGE_W - 2 * MARGIN; // A4, 2 cm margins

// inline **bold** and *italic*
function runs(text, base = {}) {
  const parts = [];
  const re = /(\*\*[^*]+\*\*|\*[^*]+\*)/g;
  let last = 0, m;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) parts.push(new TextRun({ text: text.slice(last, m.index), ...base }));
    const t = m[0];
    if (t.startsWith("**")) parts.push(new TextRun({ text: t.slice(2, -2), bold: true, ...base }));
    else parts.push(new TextRun({ text: t.slice(1, -1), italics: true, ...base }));
    last = m.index + t.length;
  }
  if (last < text.length) parts.push(new TextRun({ text: text.slice(last), ...base }));
  return parts;
}

function table(b) {
  const ncol = b.header.length;
  let widths = b.widths;
  if (!widths) widths = Array(ncol).fill(1);
  const tot = widths.reduce((a, c) => a + c, 0);
  const cw = widths.map(w => Math.floor((w / tot) * CONTENT_W));
  cw[cw.length - 1] += CONTENT_W - cw.reduce((a, c) => a + c, 0);
  const fs_ = b.small ? 15 : 17;
  const border = { style: BorderStyle.SINGLE, size: 4, color: "B0B7C3" };
  const borders = { top: border, bottom: border, left: border, right: border };
  const mkRow = (cells, header, shade) => new TableRow({
    tableHeader: header,
    children: cells.map((c, i) => new TableCell({
      width: { size: cw[i], type: WidthType.DXA },
      borders,
      shading: header ? { fill: "1F4E79", type: ShadingType.CLEAR, color: "auto" }
        : (shade ? { fill: "EEF3F9", type: ShadingType.CLEAR, color: "auto" } : undefined),
      margins: { top: 40, bottom: 40, left: 70, right: 70 },
      children: [new Paragraph({
        alignment: (i > 0 && !header && /^[-+−]?[\d.,%]+/.test(String(c))) ? AlignmentType.RIGHT : AlignmentType.LEFT,
        children: runs(String(c), { size: fs_, font: FONT, color: header ? "FFFFFF" : "000000", bold: header ? true : undefined }),
      })],
    })),
  });
  const rows = [mkRow(b.header, true, false)];
  b.rows.forEach((r, k) => rows.push(mkRow(r, false, (b.highlight || []).includes(k))));
  return new Table({ width: { size: CONTENT_W, type: WidthType.DXA }, columnWidths: cw, rows });
}

const children = [];
for (const b of C.blocks) {
  switch (b.t) {
    case "title":
      children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: b.before || 0, after: 200 },
        children: [new TextRun({ text: b.text, bold: !!b.bold, size: b.size || 24, font: FONT, color: b.color || "000000" })] }));
      break;
    case "h1": children.push(new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(b.text)] })); break;
    case "h2": children.push(new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(b.text)] })); break;
    case "h3": children.push(new Paragraph({ heading: HeadingLevel.HEADING_3, children: [new TextRun(b.text)] })); break;
    case "p": children.push(new Paragraph({ spacing: { after: 140, line: 288 }, alignment: AlignmentType.JUSTIFIED, children: runs(b.text) })); break;
    case "box":
      children.push(new Paragraph({ spacing: { before: 80, after: 160, line: 276 },
        shading: { fill: "FFF4D6", type: ShadingType.CLEAR, color: "auto" },
        border: { left: { style: BorderStyle.SINGLE, size: 18, color: "E0A800", space: 6 } },
        children: runs(b.text) }));
      break;
    case "bullets":
      b.items.forEach(it => children.push(new Paragraph({ numbering: { reference: "bul", level: 0 }, spacing: { after: 60, line: 276 }, children: runs(it) })));
      break;
    case "numbers":
      b.items.forEach(it => children.push(new Paragraph({ numbering: { reference: b.ref || "num", level: 0 }, spacing: { after: 60, line: 276 }, children: runs(it) })));
      break;
    case "eq":
      children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 60, after: 120 },
        children: [new TextRun({ text: b.text, italics: true, font: "Cambria Math", size: 21 })] }));
      break;
    case "caption":
      children.push(new Paragraph({ spacing: { before: 60, after: 200 }, children: runs(b.text, { size: 18, italics: true, color: "404040" }) }));
      break;
    case "tcaption":
      children.push(new Paragraph({ spacing: { before: 200, after: 60 }, keepNext: true, children: runs(b.text, { size: 19, bold: true, color: "1F4E79" }) }));
      break;
    case "table": children.push(table(b)); children.push(new Paragraph({ spacing: { after: 80 }, children: [] })); break;
    case "img": {
      const p = path.join(ROOT, b.path);
      const buf = fs.readFileSync(p);
      const w = buf.readUInt32BE(16), h = buf.readUInt32BE(20);
      const tw = b.width || 620; const th = Math.round(tw * h / w);
      children.push(new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 120 },
        children: [new ImageRun({ type: "png", data: buf, transformation: { width: tw, height: th },
          altText: { title: b.alt || "figure", description: b.alt || "figure", name: "fig" } })] }));
      break;
    }
    case "toc":
      children.push(new TableOfContents("Contents", { hyperlink: true, headingStyleRange: "1-2" }));
      break;
    case "toc1": children.push(new Paragraph({ spacing: { before: 80, after: 20 }, children: [new TextRun({ text: b.text, bold: true, size: 22 })] })); break;
    case "toc2": children.push(new Paragraph({ spacing: { after: 10 }, indent: { left: 400 }, children: [new TextRun({ text: b.text, size: 20 })] })); break;
    case "pb": children.push(new Paragraph({ children: [new PageBreak()] })); break;
    case "ref":
      children.push(new Paragraph({ spacing: { after: 90, line: 264 }, indent: { left: 400, hanging: 400 }, children: runs(b.text, { size: 20 }) }));
      break;
  }
}

const doc = new Document({
  creator: "Group", title: C.title,
  styles: {
    default: { document: { run: { font: FONT, size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 32, bold: true, font: FONT, color: "1F4E79" }, paragraph: { spacing: { before: 360, after: 160 }, outlineLevel: 0, keepNext: true } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 26, bold: true, font: FONT, color: "2E5C8A" }, paragraph: { spacing: { before: 240, after: 120 }, outlineLevel: 1, keepNext: true } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 23, bold: true, font: FONT, color: "404040" }, paragraph: { spacing: { before: 180, after: 80 }, outlineLevel: 2, keepNext: true } },
    ],
  },
  numbering: { config: [
    { reference: "bul", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 540, hanging: 270 } } } }] },
    ...["num", "num2", "num3", "num4", "num5", "num6", "num7", "num8"].map(r => ({ reference: r, levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.",
      alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 300 } } } }] })),
  ] },
  sections: [{
    properties: { page: { size: { width: PAGE_W, height: 16838 }, margin: { top: MARGIN, bottom: MARGIN, left: MARGIN, right: MARGIN } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
      children: [new TextRun({ children: [PageNumber.CURRENT], size: 18, color: "808080" })] })] }) },
    children,
  }],
});
Packer.toBuffer(doc).then(b => { fs.writeFileSync(out, b); console.log("wrote", out); });
