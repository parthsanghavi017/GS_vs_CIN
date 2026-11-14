import os
import docx
from docx.shared import Inches, Pt, RGBColor, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

doc = docx.Document()

# Page Margins: 1.7 cm on all sides (Exact template requirement: Top=1.7cm, Bottom=1.7cm, Left=1.7cm, Right=1.7cm)
for section in doc.sections:
    section.top_margin = Cm(1.7)
    section.bottom_margin = Cm(1.7)
    section.left_margin = Cm(1.7)
    section.right_margin = Cm(1.7)
    section.page_width = Cm(21.0)
    section.page_height = Cm(29.7)
    
    # Configure Header
    header = section.header
    header.is_linked_to_previous = False
    hp = header.paragraphs[0]
    hp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    hp.paragraph_format.space_before = Pt(0)
    hp.paragraph_format.space_after = Pt(2)
    hrun1 = hp.add_run("Bengaluru Tech Summit 2026\n")
    hrun1.font.name = "Times New Roman"
    hrun1.font.size = Pt(11)
    hrun1.font.bold = True
    hrun2 = hp.add_run("November 17-19, 2026, Bangalore International Exhibition Center")
    hrun2.font.name = "Times New Roman"
    hrun2.font.size = Pt(9.5)
    
    # Bottom page number in footer (Bottom centred: template requirement)
    footer = section.footer
    fp = footer.paragraphs[0]
    fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    fp.paragraph_format.space_before = Pt(4)
    fp.paragraph_format.space_after = Pt(0)
    frun = fp.add_run()
    frun.font.name = "Times New Roman"
    frun.font.size = Pt(10)
    f_p = fp._p
    f_pPr = f_p.get_or_add_pPr()
    fldSimple = OxmlElement('w:fldSimple')
    fldSimple.set(qn('w:instr'), 'PAGE')
    fp._p.append(fldSimple)

def set_run_font(run, name="Times New Roman", size=10, bold=False, italic=False, color=None):
    run.font.name = name
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    if color:
        run.font.color.rgb = color

def add_hr(doc):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(4)
    pBdr = parse_xml(r'<w:pBdr xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
                     r'<w:bottom w:val="single" w:sz="6" w:space="1" w:color="000000"/>'
                     r'</w:pBdr>')
    p._p.get_or_add_pPr().append(pBdr)

# Header line divider
add_hr(doc)

# Title (Font 14, Times New Roman, Bold, Centered)
p_title = doc.add_paragraph()
p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
p_title.paragraph_format.space_before = Pt(2)
p_title.paragraph_format.space_after = Pt(4)
run_title = p_title.add_run("Machine Learning-Driven Multi-Omics Classification of Genomically Stable and Chromosomal Instability Gastric Adenocarcinoma Without Copy-Number Profiling")
set_run_font(run_title, size=14, bold=True)

# Authors (Font 12, Times New Roman, Centered)
p_authors = doc.add_paragraph()
p_authors.alignment = WD_ALIGN_PARAGRAPH.CENTER
p_authors.paragraph_format.space_before = Pt(2)
p_authors.paragraph_format.space_after = Pt(3)
run_auth = p_authors.add_run("Author One1*, Author Two2, Author Three1")
set_run_font(run_auth, size=12)

# Affiliations (Font 10, Times New Roman, Centered)
p_aff1 = doc.add_paragraph()
p_aff1.alignment = WD_ALIGN_PARAGRAPH.CENTER
p_aff1.paragraph_format.space_before = Pt(0)
p_aff1.paragraph_format.space_after = Pt(1)
run_aff1 = p_aff1.add_run("1Department of Biotechnology / Computational Biology, Institution / University Name, City, State, India")
set_run_font(run_aff1, size=10)

p_aff2 = doc.add_paragraph()
p_aff2.alignment = WD_ALIGN_PARAGRAPH.CENTER
p_aff2.paragraph_format.space_before = Pt(0)
p_aff2.paragraph_format.space_after = Pt(1)
run_aff2 = p_aff2.add_run("2Department of Bioinformatics / Oncology, Partner Institution / Hospital Name, City, State, India")
set_run_font(run_aff2, size=10)

# Email (Font 10, Times New Roman, Centered)
p_email = doc.add_paragraph()
p_email.alignment = WD_ALIGN_PARAGRAPH.CENTER
p_email.paragraph_format.space_before = Pt(0)
p_email.paragraph_format.space_after = Pt(3)
run_email = p_email.add_run("E-mail addresses: author1@institution.edu, author2@institution.edu, author3@institution.edu")
set_run_font(run_email, size=10)

add_hr(doc)

# 1. Introduction:
p_intro_h = doc.add_paragraph()
p_intro_h.paragraph_format.space_before = Pt(4)
p_intro_h.paragraph_format.space_after = Pt(2)
run_h1 = p_intro_h.add_run("1. Introduction:")
set_run_font(run_h1, size=10, bold=True)

p_intro = doc.add_paragraph()
p_intro.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
p_intro.paragraph_format.space_before = Pt(0)
p_intro.paragraph_format.space_after = Pt(5)
p_intro.paragraph_format.line_spacing = 1.12
run_itxt = p_intro.add_run(
    "Gastric adenocarcinoma (STAD) displays marked molecular and clinical heterogeneity. "
    "In the non-hypermutated disease spectrum, tumors are categorized into Genomically Stable (GS) "
    "and Chromosomal Instability (CIN) subtypes [1]. Clinically, GS tumors are associated with "
    "substantially poorer overall survival and marked resistance to adjuvant chemotherapy compared to CIN [2]. "
    "Furthermore, GS tumors are strongly enriched for diffuse-type histology under the Laurén classification [3], "
    "whereas CIN tumors are predominantly intestinal. Although CIN is defined by somatic aneuploidy, standard "
    "copy-number assays require high amounts of intact DNA, incurring prohibitive costs and high failure rates in "
    "degraded formalin-fixed paraffin-embedded (FFPE) biopsies. To overcome this translational bottleneck, we developed "
    "a non-invasive multi-omics machine learning framework that stratifies GS and CIN using mRNA expression, transcription "
    "factor (TF) activity, and somatic mutational signatures without requiring copy-number data."
)
set_run_font(run_itxt, size=10)

# 2. Material and Methods:
p_mm_h = doc.add_paragraph()
p_mm_h.paragraph_format.space_before = Pt(4)
p_mm_h.paragraph_format.space_after = Pt(2)
run_h2 = p_mm_h.add_run("2. Material and Methods:")
set_run_font(run_h2, size=10, bold=True)

p_mm = doc.add_paragraph()
p_mm.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
p_mm.paragraph_format.space_before = Pt(0)
p_mm.paragraph_format.space_after = Pt(5)
p_mm.paragraph_format.line_spacing = 1.12
run_mmtxt = p_mm.add_run(
    "Clinical and molecular profiles of 205 curated TCGA-STAD patients (147 CIN, 58 GS; excluding EBV and MSI) "
    "were investigated. Phase 1 comprised clinicopathological characterization and age-stratified Kaplan-Meier overall survival "
    "(OS) estimation. Phase 2 extracted multi-modal molecular features: (i) 46 differentially expressed genes via PyDESeq2 "
    "negative binomial Wald testing, (ii) 30 TF regulon activities inferred using decoupleR and DoRothEA univariate linear models, "
    "and (iii) 5 COSMIC Single Base Substitution (SBS) mutational signatures deconvolved via non-negative matrix factorization. "
    "In the predictive modeling phase, 81 integrated features across 177 patients were evaluated across five algorithms "
    "(Random Forest, SVM, LightGBM, XGBoost, Logistic Regression) using SMOTE-balanced 10-fold cross-validation on 80% training "
    "data and evaluated on an independent holdout test cohort (N = 36; 25 CIN, 11 GS)."
)
set_run_font(run_mmtxt, size=10)

# 3. Significant Results and Discussion
p_res_h = doc.add_paragraph()
p_res_h.paragraph_format.space_before = Pt(4)
p_res_h.paragraph_format.space_after = Pt(2)
run_h3 = p_res_h.add_run("3. Significant Results and Discussion")
set_run_font(run_h3, size=10, bold=True)

p_res1 = doc.add_paragraph()
p_res1.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
p_res1.paragraph_format.space_before = Pt(0)
p_res1.paragraph_format.space_after = Pt(4)
p_res1.paragraph_format.line_spacing = 1.12
run_res1 = p_res1.add_run(
    "In Phase 1, GS patients presented at a significantly younger age (median 59.4 vs. 68.4 years, p < 0.0001) and "
    "exhibited diffuse histology enrichment (69.0% vs. 12.2%, p < 0.0001). Age-stratified survival analysis (age <= 70 years) "
    "revealed severe 5-year OS deficits in GS relative to CIN (23.1% vs. 68.1%, log-rank p = 0.021). "
    "In Phase 2, CIN tumors displayed pronounced upregulation of mitotic spindle regulators (UBE2C, AURKA), elevated E2F "
    "transcription factor activity (E2F2, E2F3, TFDP1), and significant enrichment in APOBEC cytidine deaminase mutagenesis "
    "(SBS2/13 mean exposure: 0.221 CIN vs. 0.104 GS, p = 1.37e-7). Conversely, GS tumors showed elevated GLI1 activity and "
    "diffuse-associated cytoskeleton expression (SEPTIN6, SH3BP5)."
)
set_run_font(run_res1, size=10)

# Explicit Page Break to start Page 2 cleanly with Table 1 and Figure 1
doc.add_page_break()

# Page 2: Table 1 Caption Centered Above Table
p_tbl_cap = doc.add_paragraph()
p_tbl_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
p_tbl_cap.paragraph_format.space_before = Pt(0)
p_tbl_cap.paragraph_format.space_after = Pt(3)
run_tcap = p_tbl_cap.add_run("Table 1: Performance of copy-number-free machine learning classifiers (81 features)")
set_run_font(run_tcap, size=9.5, bold=True)

# Table 1
table = doc.add_table(rows=6, cols=6)
table.alignment = WD_TABLE_ALIGNMENT.CENTER
headers = ["Classifier", "10-Fold CV AUC", "Test ROC-AUC", "Test PR-AUC", "Balanced Acc.", "Test F1-Score"]
data = [
    ["Random Forest", "0.891", "0.975", "0.990", "0.798", "0.906"],
    ["SVM", "0.890", "0.945", "0.976", "0.844", "0.923"],
    ["LightGBM", "0.876", "0.949", "0.980", "0.758", "0.863"],
    ["XGBoost", "0.865", "0.920", "0.964", "0.804", "0.880"],
    ["Logistic Reg.", "0.847", "0.913", "0.963", "0.778", "0.885"],
]

for col_idx, h in enumerate(headers):
    cell = table.cell(0, col_idx)
    cell.text = h
    cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
    for r in cell.paragraphs[0].runs:
        set_run_font(r, size=9, bold=True)
    shading = parse_xml(r'<w:shd {} w:fill="F2F4F7"/>'.format(nsdecls('w')))
    cell._tc.get_or_add_tcPr().append(shading)

for row_idx, row_vals in enumerate(data):
    for col_idx, val in enumerate(row_vals):
        cell = table.cell(row_idx + 1, col_idx)
        cell.text = val
        cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER if col_idx > 0 else WD_ALIGN_PARAGRAPH.LEFT
        for r in cell.paragraphs[0].runs:
            bold_flag = True if (row_idx == 0 and col_idx in [2, 3, 5]) or (row_idx == 1 and col_idx in [4, 5]) else False
            set_run_font(r, size=8.5, bold=bold_flag)

# Add clean borders to Table 1
tblBorders = parse_xml(r'<w:tblBorders {} >'
                       r'<w:top w:val="single" w:sz="6" w:space="0" w:color="000000"/>'
                       r'<w:bottom w:val="single" w:sz="6" w:space="0" w:color="000000"/>'
                       r'<w:left w:val="none"/>'
                       r'<w:right w:val="none"/>'
                       r'<w:insideH w:val="single" w:sz="4" w:space="0" w:color="D0D5DD"/>'
                       r'<w:insideV w:val="none"/>'
                       r'</w:tblBorders>'.format(nsdecls('w')))
table._tbl.tblPr.append(tblBorders)

for row in table.rows:
    for cell in row.cells:
        tcPr = cell._tc.get_or_add_tcPr()
        tcMar = parse_xml(r'<w:tcMar {}><w:top w:w="50" w:type="dxa"/><w:bottom w:w="50" w:type="dxa"/><w:left w:w="90" w:type="dxa"/><w:right w:w="90" w:type="dxa"/></w:tcMar>'.format(nsdecls('w')))
        tcPr.append(tcMar)

# Figure 1
fig_rel_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "phase4_rna_tf_mutsig", "figures", "phase4_roc_pr.png")
fig_path = fig_rel_path if os.path.exists(fig_rel_path) else "/home/anusha/Projects/Tech-summit/GS_CIN_STAD/GS_vs_CIN-main/phase4_rna_tf_mutsig/figures/phase4_roc_pr.png"
if os.path.exists(fig_path):
    p_fig = doc.add_paragraph()
    p_fig.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_fig.paragraph_format.space_before = Pt(4)
    p_fig.paragraph_format.space_after = Pt(2)
    run_fig = p_fig.add_run()
    run_fig.add_picture(fig_path, width=Inches(5.6))

p_fig_cap = doc.add_paragraph()
p_fig_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
p_fig_cap.paragraph_format.space_before = Pt(1)
p_fig_cap.paragraph_format.space_after = Pt(4)
run_fcap = p_fig_cap.add_run("Figure 1: Receiver operating characteristic (ROC) and Precision-Recall (PR) curves on holdout test cohort (N = 36)")
set_run_font(run_fcap, size=9, italic=True)

# Predictive results discussion
p_res2 = doc.add_paragraph()
p_res2.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
p_res2.paragraph_format.space_before = Pt(0)
p_res2.paragraph_format.space_after = Pt(4)
p_res2.paragraph_format.line_spacing = 1.12
run_res2 = p_res2.add_run(
    "In predictive modeling (Table 1, Figure 1), the copy-number-free classifier achieved outstanding diagnostic accuracy. "
    "Random Forest achieved holdout Test ROC-AUC of 0.975, PR-AUC of 0.990, and F1-score of 0.906, correctly identifying 24 of 25 "
    "CIN patients (96% sensitivity). SVM demonstrated balanced accuracy of 0.844 and F1-score of 0.923. Crucially, integrating "
    "mutational signatures with mRNA and TF regulons boosted holdout ROC-AUC from 0.941 to 0.975, demonstrating that orthogonal "
    "mutational and transcriptomic signatures effectively surrogate for DNA copy-number alterations."
)
set_run_font(run_res2, size=10)

# 4. Conclusions:
p_concl_h = doc.add_paragraph()
p_concl_h.paragraph_format.space_before = Pt(4)
p_concl_h.paragraph_format.space_after = Pt(2)
run_h4 = p_concl_h.add_run("4. Conclusions:")
set_run_font(run_h4, size=10, bold=True)

p_concl = doc.add_paragraph()
p_concl.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
p_concl.paragraph_format.space_before = Pt(0)
p_concl.paragraph_format.space_after = Pt(4)
p_concl.paragraph_format.line_spacing = 1.12
run_concltxt = p_concl.add_run(
    "This study demonstrates that an 81-feature multi-omics classifier integrating mRNA transcripts, TF regulons, "
    "and COSMIC mutational signatures achieves near-ceiling diagnostic accuracy (ROC-AUC = 0.975, PR-AUC = 0.990) "
    "in distinguishing GS from CIN gastric adenocarcinoma. By eliminating the necessity for costly DNA copy-number assays, "
    "this pipeline offers a highly robust, cost-effective, and clinically viable tool for patient stratification and therapy selection."
)
set_run_font(run_concltxt, size=10)

# References
p_ref_h = doc.add_paragraph()
p_ref_h.paragraph_format.space_before = Pt(4)
p_ref_h.paragraph_format.space_after = Pt(1)
run_href = p_ref_h.add_run("References")
set_run_font(run_href, size=10, bold=True)

ref_p = doc.add_paragraph()
ref_p.paragraph_format.space_before = Pt(0)
ref_p.paragraph_format.space_after = Pt(2)
ref_p.paragraph_format.line_spacing = 1.1
r1 = ref_p.add_run("[1] The Cancer Genome Atlas Research Network, Nature, 513 (7517) 2014, 202-209.\n")
set_run_font(r1, size=9)
r2 = ref_p.add_run("[2] B. H. Sohn, J. E. Hwang, H. J. Jang, et al., Clin. Cancer Res., 23 (15) 2017, 4441-4449.\n")
set_run_font(r2, size=9)
r3 = ref_p.add_run("[3] P. Laurén, Acta Pathol. Microbiol. Scand., 64 (1) 1965, 31-49.")
set_run_font(r3, size=9)

out_path = "/home/anusha/Projects/Tech-summit/GS_CIN_STAD/Bengaluru_Tech_Summit_2026_Abstract_GS_vs_CIN.docx"
doc.save(out_path)
print(f"Saved refined DOCX to {out_path}")
