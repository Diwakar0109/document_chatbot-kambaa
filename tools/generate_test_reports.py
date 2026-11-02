"""Generate complex synthetic hospital test reports for testing Dr. Asha.

Creates (in ../test_reports/):
  1. Metabolic_CBC_Lipid_Panel_Report.docx  — 45yo male, T2DM + hypertension
  2. Thyroid_VitaminD_Iron_Panel_Report.docx — 32yo female, fatigue workup
  3. Lab_Trends_Workbook.xlsx               — longitudinal lab trends

All data is synthetic. Run from anywhere:  python tools/generate_test_reports.py
"""

from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt, RGBColor
import pandas as pd

OUT = Path(__file__).resolve().parent.parent / "test_reports"
OUT.mkdir(exist_ok=True)

RED = RGBColor(0xB9, 0x1C, 0x1C)


def banner(doc, lines):
    table = doc.add_table(rows=0, cols=2)
    table.style = "Light Grid Accent 1"
    for k, v in lines:
        row = table.add_row().cells
        row[0].text = k
        row[1].text = v
    return table


def lab_table(doc, title, rows, note=None):
    doc.add_heading(title, level=2)
    t = doc.add_table(rows=1, cols=4)
    t.style = "Light Grid Accent 1"
    hdr = t.rows[0].cells
    for i, h in enumerate(["Test", "Result", "Flag", "Reference Range"]):
        hdr[i].text = h
        for run in hdr[i].paragraphs[0].runs:
            run.bold = True
    for name, result, flag, ref in rows:
        cells = t.add_row().cells
        cells[0].text, cells[1].text, cells[2].text, cells[3].text = name, result, flag, ref
        if flag:
            for run in cells[2].paragraphs[0].runs:
                run.bold = True
                run.font.color.rgb = RED
    if note:
        doc.add_paragraph(note).runs and None
        doc.paragraphs[-1].add_run(f" {note}").italic = True
    doc.add_paragraph()


def meta_report():
    doc = Document()
    h = doc.add_heading("ST. MARYA MEMORIAL HOSPITAL — DEPARTMENT OF LABORATORY MEDICINE", level=0)
    h.alignment = WD_ALIGN_PARAGRAPH.CENTER

    banner(doc, [
        ("Patient Name", "Mr. Rajesh K. (Synthetic Test Data)"),
        ("Age / Sex", "45 Years / Male"),
        ("Patient ID", "SYN-2024-118842"),
        ("Referred By", "Dr. Anita Verma, Internal Medicine"),
        ("Collection / Reporting", "07:35 AM / 01:20 PM, 14 Mar 2024"),
        ("Specimen", "Fasting venous blood (12h fast), random urine"),
    ])

    doc.add_heading("Clinical History", level=1)
    doc.add_paragraph(
        "Known case of type 2 diabetes mellitus (6 years) on metformin 500 mg BD and glimepiride 1 mg OD. "
        "Hypertension (4 years) on telmisartan 40 mg OD. Presents for quarterly metabolic review. "
        "Reports occasional nocturnal leg cramps and increased fatigue over the past month. "
        "Denies chest pain, dysuria, polyuria or visual disturbances."
    )

    lab_table(doc, "Complete Metabolic Panel (CMP)", [
        ("Fasting Blood Glucose", "168 mg/dL", "H", "70 – 99"),
        ("HbA1c (NGSP-aligned)", "8.4 %", "H", "4.0 – 5.6 (goal < 7.0 for T2DM)"),
        ("Urea", "38 mg/dL", "", "17 – 43"),
        ("Creatinine", "1.12 mg/dL", "", "0.70 – 1.30"),
        ("eGFR (CKD-EPI)", "78 mL/min/1.73m²", "", "> 90"),
        ("Sodium", "139 mmol/L", "", "136 – 145"),
        ("Potassium", "4.8 mmol/L", "", "3.5 – 5.1"),
        ("Chloride", "103 mmol/L", "", "98 – 107"),
        ("Bicarbonate", "24 mmol/L", "", "22 – 29"),
        ("Calcium", "9.1 mg/dL", "", "8.6 – 10.3"),
        ("Total Bilirubin", "0.9 mg/dL", "", "0.3 – 1.2"),
        ("AST (SGOT)", "42 U/L", "H", "10 – 40"),
        ("ALT (SGPT)", "55 U/L", "H", "10 – 49"),
        ("ALP", "98 U/L", "", "40 – 129"),
        ("Total Protein", "7.2 g/dL", "", "6.4 – 8.3"),
        ("Albumin", "4.3 g/dL", "", "3.5 – 5.2"),
    ])

    lab_table(doc, "Complete Blood Count (CBC)", [
        ("Hemoglobin", "11.8 g/dL", "L", "13.0 – 17.0"),
        ("Hematocrit", "36.9 %", "L", "40 – 52"),
        ("RBC Count", "4.4 mill/mm³", "", "4.5 – 5.9"),
        ("MCV", "74 fL", "L", "83 – 101"),
        ("MCH", "26.8 pg", "L", "27 – 32"),
        ("MCHC", "31.9 g/dL", "L", "32 – 36"),
        ("RDW-CV", "16.4 %", "H", "11.6 – 14.0"),
        ("Total Leukocyte Count", "8.9 thou/mm³", "", "4.0 – 10.0"),
        ("Platelet Count", "262 thou/mm³", "", "150 – 410"),
        ("ESR", "22 mm/hr", "H", "0 – 15"),
    ])

    lab_table(doc, "Lipid Profile", [
        ("Total Cholesterol", "238 mg/dL", "H", "< 200"),
        ("LDL Cholesterol (calc)", "158 mg/dL", "H", "< 100 (diabetic goal < 70)"),
        ("HDL Cholesterol", "34 mg/dL", "L", "> 40"),
        ("Triglycerides", "242 mg/dL", "H", "< 150"),
        ("VLDL Cholesterol", "48 mg/dL", "H", "5 – 40"),
        ("Non-HDL Cholesterol", "204 mg/dL", "H", "< 130"),
        ("Total Chol / HDL Ratio", "7.0", "H", "< 5.0"),
    ])

    lab_table(doc, "Urinalysis", [
        ("Colour / Appearance", "Yellow / Clear", "", "—"),
        ("Protein (Albumin)", "Trace", "", "Nil"),
        ("Urine Albumin-Creatinine Ratio", "62 mg/g", "H", "< 30"),
        ("Glucose", "Present", "H", "Absent"),
        ("Ketones", "Absent", "", "Absent"),
        ("Pus Cells", "2-3 /HPF", "", "0 – 4"),
        ("RBCs", "Nil", "", "0 – 2"),
    ])

    doc.add_heading("Impression", level=1)
    for item in [
        "Glycemic control suboptimal: HbA1c 8.4% and fasting glucose 168 mg/dL exceed ADA targets "
        "(goal HbA1c < 7.0%); medication review advised.",
        "Microcytic hypochromic anemia (Hb 11.8, MCV 74) with raised RDW — iron-deficiency pattern; "
        "iron studies and occult blood evaluation recommended.",
        "Borderline hepatocellular enzyme elevation (AST 42, ALT 55) — plausible metabolic/diabetic "
        "hepatic steatosis; correlate clinically, repeat in 6-8 weeks.",
        "Dyslipidemia with low HDL (34) and elevated triglycerides (242) — intensify statin per "
        "ASCVD risk assessment.",
        "Moderately increased albuminuria (UACR 62 mg/g) consistent with early diabetic kidney "
        "involvement; nephrology follow-up and ACEi/ARB optimization advised.",
        "eGFR 78: mild reduction — recheck in 3 months with UACR.",
    ]:
        doc.add_paragraph(item, style="List Bullet")

    doc.add_heading("Recommendations", level=1)
    doc.add_paragraph(
        "Repeat HbA1c in 3 months after therapy adjustment. Iron studies (serum iron, TIBC, ferritin) "
        "and stool occult blood. Fasting lipid profile post-statin titration. Renal diet counselling; "
        "avoid NSAIDs. Annual dilated eye examination due for review. Consider continuous glucose "
        "monitoring if adherence to SMBG is poor."
    )
    p = doc.add_paragraph()
    run = p.add_run("This is a SYNTHETIC document generated for software testing. Not a real patient report.")
    run.italic = True
    run.font.size = Pt(9)

    path = OUT / "Metabolic_CBC_Lipid_Panel_Report.docx"
    doc.save(path)
    return path


def thyroid_report():
    doc = Document()
    h = doc.add_heading("CITY CARE DIAGNOSTICS — SPECIALIZED PATHOLOGY LABORATORY", level=0)
    h.alignment = WD_ALIGN_PARAGRAPH.CENTER

    banner(doc, [
        ("Patient Name", "Ms. Meera S. (Synthetic Test Data)"),
        ("Age / Sex", "32 Years / Female"),
        ("Patient ID", "SYN-2024-092217"),
        ("Referred By", "Dr. Kavita Rao, Endocrinology"),
        ("Collection / Reporting", "09:10 AM / 04:45 PM, 02 Apr 2024"),
        ("Specimen", "Serum (clot activator), morning sample"),
    ])

    doc.add_heading("Clinical History", level=1)
    doc.add_paragraph(
        "Progressive fatigue, palpitations, heat intolerance and unexplained weight loss (5 kg in 3 months) "
        "despite normal appetite. Irregular menstrual cycles for 2 months. Not pregnant (UPT negative). "
        "No medications other than oral contraceptive pills. Family history: mother hypothyroid."
    )

    lab_table(doc, "Thyroid Function Panel", [
        ("TSH (ultrasensitive)", "0.011 µIU/mL", "L", "0.45 – 4.50"),
        ("Free T4", "2.31 ng/dL", "H", "0.82 – 1.77"),
        ("Free T3", "5.9 pg/mL", "H", "2.0 – 4.4"),
        ("Anti-TPO Antibodies", "18 IU/mL", "", "< 35"),
        ("Anti-TSH Receptor (TRAb)", "4.8 IU/L", "H", "< 1.75"),
        ("Thyroglobulin", "86 ng/mL", "H", "3 – 40"),
    ])

    lab_table(doc, "Vitamin & Bone Panel", [
        ("25-OH Vitamin D (total)", "12.4 ng/mL", "L", "30 – 100 (deficiency < 20)"),
        ("Calcium (corrected)", "9.4 mg/dL", "", "8.6 – 10.3"),
        ("Phosphorus", "3.9 mg/dL", "", "2.5 – 4.5"),
        ("Intact PTH", "68 pg/mL", "H", "15 – 65"),
        ("Magnesium", "2.0 mg/dL", "", "1.6 – 2.6"),
    ])

    lab_table(doc, "Iron Studies & Hemogram", [
        ("Hemoglobin", "10.9 g/dL", "L", "12.0 – 15.5"),
        ("MCV", "71 fL", "L", "80 – 100"),
        ("Serum Iron", "28 µg/dL", "L", "50 – 170"),
        ("TIBC", "462 µg/dL", "H", "250 – 425"),
        ("Transferrin Saturation", "6 %", "L", "20 – 50"),
        ("Serum Ferritin", "8 ng/mL", "L", "15 – 150"),
    ])

    lab_table(doc, "Reproductive Hormone Panel", [
        ("LH", "4.2 mIU/mL", "", "Follicular 2.4 – 12.6"),
        ("FSH", "5.6 mIU/mL", "", "Follicular 3.5 – 12.5"),
        ("Estradiol (E2)", "88 pg/mL", "", "Follicular 12 – 166"),
        ("Prolactin", "31 ng/mL", "H", "4.8 – 23.3"),
        ("β-hCG (qualitative)", "Negative", "", "Negative"),
    ])

    doc.add_heading("Impression", level=1)
    for item in [
        "Biochemical thyrotoxicosis with suppressed TSH (0.011) and elevated Free T4/T3; positive TRAb "
        "(4.8) is consistent with Graves' disease. Anti-TPO negative. Endocrinology referral for "
        "treatment planning (anti-thyroid medication vs radioiodine); thyroid uptake scan advised.",
        "Severe iron-deficiency anemia (ferritin 8, TSAT 6%) in a menstruating woman — investigate "
        "menstrual blood loss and consider upper/lower GI evaluation if no gynecologic cause; iron "
        "replacement indicated.",
        "Severe vitamin D deficiency (12.4 ng/mL) with secondary hyperparathyroidism (PTH 68) — "
        "cholecalciferol loading followed by maintenance, per local protocol.",
        "Mildly elevated prolactin (31) — can be seen in primary hypothyroidism/stress; repeat after "
        "thyroid management, macroprolactin if persistent.",
    ]:
        doc.add_paragraph(item, style="List Bullet")

    doc.add_heading("Recommendations", level=1)
    doc.add_paragraph(
        "Urgent endocrinology review (within 1-2 weeks) for thyrotoxicosis. Beta-blocker symptomatic "
        "control per treating physician. Oral iron with vitamin C on alternate days, away from tea/coffee. "
        "Vitamin D 60,000 IU weekly × 8 weeks then monthly maintenance. Repeat TSH/FT4 in 4-6 weeks; CBC "
        "and ferritin in 4 weeks. Counsel on warning signs of thyroid storm."
    )
    p = doc.add_paragraph()
    run = p.add_run("This is a SYNTHETIC document generated for software testing. Not a real patient report.")
    run.italic = True
    run.font.size = Pt(9)

    path = OUT / "Thyroid_VitaminD_Iron_Panel_Report.docx"
    doc.save(path)
    return path


def trends_workbook():
    path = OUT / "Lab_Trends_Workbook.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame({
            "Date": ["2022-06", "2022-12", "2023-06", "2023-12", "2024-03"],
            "HbA1c (%)": [7.1, 7.6, 7.9, 8.1, 8.4],
            "Fasting Glucose (mg/dL)": [128, 141, 149, 158, 168],
            "Post-meal Glucose (mg/dL)": [182, 196, 205, 214, 231],
            "Goal": ["< 7.0"] * 5,
        }).to_excel(writer, sheet_name="HbA1c Trend", index=False)

        pd.DataFrame({
            "Date": ["2022-06", "2022-12", "2023-06", "2023-12", "2024-03"],
            "Total Cholesterol": [206, 214, 221, 230, 238],
            "LDL": [128, 136, 144, 151, 158],
            "HDL": [41, 39, 38, 36, 34],
            "Triglycerides": [168, 183, 201, 224, 242],
            "On Statin": ["No", "No", "No", "No", "No"],
        }).to_excel(writer, sheet_name="Lipid Trend", index=False)

        pd.DataFrame({
            "Month": ["Oct", "Nov", "Dec", "Jan", "Feb", "Mar"],
            "Systolic (mmHg)": [138, 142, 136, 144, 148, 150],
            "Diastolic (mmHg)": [86, 88, 84, 90, 92, 94],
            "Heart Rate": [76, 78, 74, 80, 82, 84],
            "Missed Doses (Telmisartan)": [0, 1, 0, 2, 3, 2],
        }).to_excel(writer, sheet_name="BP Log", index=False)

        pd.DataFrame({
            "Parameter": ["Weight (kg)", "BMI", "Waist (cm)", "Steps/day (avg)", "Sleep (hrs)"],
            "2023-03": [84, 28.4, 98, 4200, 6.2],
            "2023-09": [86, 29.1, 100, 3800, 5.9],
            "2024-03": [88, 29.8, 103, 3100, 5.4],
        }).to_excel(writer, sheet_name="Lifestyle Metrics", index=False)
    return path


if __name__ == "__main__":
    for fn in (meta_report, thyroid_report, trends_workbook):
        print("Created:", fn())
