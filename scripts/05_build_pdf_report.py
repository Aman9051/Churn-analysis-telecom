from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib import colors
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                 Table, TableStyle, ListFlowable, ListItem)

OUT = "outputs"
doc = SimpleDocTemplate(f"{OUT}/Churn_Analysis_Report.pdf", pagesize=A4,
                         topMargin=1.4*cm, bottomMargin=1.4*cm,
                         leftMargin=1.8*cm, rightMargin=1.8*cm)

styles = getSampleStyleSheet()
title_style = ParagraphStyle("TitleStyle", parent=styles["Title"], fontSize=17,
                              spaceAfter=4, textColor=colors.HexColor("#1f3864"))
h2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=11.5, spaceBefore=8,
                     spaceAfter=3, textColor=colors.HexColor("#1f3864"))
body = ParagraphStyle("Body", parent=styles["Normal"], fontSize=9.3, leading=12.5,
                       alignment=TA_JUSTIFY, spaceAfter=4)
small = ParagraphStyle("Small", parent=styles["Normal"], fontSize=8.5, leading=11,
                        textColor=colors.HexColor("#444444"))

story = []

story.append(Paragraph("Customer Churn Analysis for Telecom Industry", title_style))
story.append(Paragraph("Data Analyst Internship Project Report &middot; "
                        "Dataset: IBM Telco Customer Churn (Kaggle)", small))
story.append(Spacer(1, 8))

story.append(Paragraph("Abstract", h2))
story.append(Paragraph(
"Customer churn is one of the costliest problems in the telecom industry, where acquiring a new "
"subscriber is far more expensive than retaining an existing one. This project analyzes the IBM "
"Telco Customer Churn dataset (7,043 customers, 21 attributes, 26.5% churn rate) to identify the "
"contractual and service-related factors driving churn, builds a machine-learning classifier to "
"predict at-risk customers, and segments the customer base into actionable retention groups.", body))

story.append(Paragraph("Introduction", h2))
story.append(Paragraph(
"Telecom companies operate in a highly saturated, price-sensitive market where customer attrition "
"directly erodes recurring revenue. Each customer record captures demographics, contract type, "
"billing, subscribed services (internet, online security, tech support, streaming), and tenure. "
"The goal is to move from reactive churn reporting to a proactive, data-driven retention strategy "
"by quantifying which factors matter most and which customers need attention first.", body))

story.append(Paragraph("Tools Used", h2))
story.append(Paragraph(
"<b>Python</b> (Pandas, NumPy) for data cleaning and feature preparation &nbsp;|&nbsp; "
"<b>SQL</b> (SQLite) for aggregation queries on contract, tenure, charges, and services &nbsp;|&nbsp; "
"<b>Scikit-learn</b> for Logistic Regression and Random Forest classification models &nbsp;|&nbsp; "
"<b>Permutation Importance</b> (ELI5-style explainability) to interpret model drivers &nbsp;|&nbsp; "
"<b>Matplotlib / Seaborn</b> for exploratory visualizations.", body))

story.append(Paragraph("Steps Involved in Building the Project", h2))
steps = [
    "Cleaned the raw Kaggle dataset: converted <i>TotalCharges</i> to numeric and imputed 11 blank "
    "values for brand-new (tenure = 0) customers, standardized column names.",
    "Loaded the cleaned data into a SQLite database and ran SQL aggregation queries: churn rate by "
    "contract, tenure bucket, internet service, payment method, and tech support.",
    "Performed EDA to visualize churn patterns across contract type, tenure, and monthly charges.",
    "Encoded categorical variables and trained two binary classifiers (Logistic Regression baseline, "
    "Random Forest) on a 75/25 train-test split.",
    "Evaluated models using accuracy, precision, recall, F1-score, ROC-AUC, and a confusion matrix.",
    "Applied permutation importance (ELI5-style) to rank the top drivers of churn for explainability.",
    "Scored every customer with a churn probability and grouped them into three segments: "
    "<b>At Risk</b>, <b>Loyal</b>, and <b>Dormant</b> for targeted retention action.",
]
story.append(ListFlowable(
    [ListItem(Paragraph(s, body), bulletColor=colors.HexColor("#1f3864")) for s in steps],
    bulletType="bullet", start="circle", leftIndent=12
))

story.append(Paragraph("Key Results", h2))
results_table = Table([
    ["Model", "Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"],
    ["Logistic Regression", "0.747", "0.515", "0.801", "0.627", "0.844"],
    ["Random Forest", "0.776", "0.564", "0.690", "0.620", "0.839"],
], colWidths=[3.6*cm, 2*cm, 2*cm, 2*cm, 2*cm, 2*cm])
results_table.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f3864")),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTSIZE", (0, 0), (-1, -1), 8.3),
    ("ALIGN", (1, 0), (-1, -1), "CENTER"),
    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f5fa")]),
    ("TOPPADDING", (0, 0), (-1, -1), 3),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
]))
story.append(results_table)
story.append(Spacer(1, 4))
story.append(Paragraph(
"<b>Top churn drivers (permutation importance):</b> contract type (dominant factor), tenure, and "
"online security enrollment. <b>SQL aggregation confirmed:</b> month-to-month customers churn at "
"42.7% vs. 11.3% (one-year) and 2.8% (two-year); fiber-optic customers churn at 41.9% vs. 19.0% "
"(DSL); electronic check payers churn at 45.3%, the highest of any payment method; churn is highest "
"in the first 12 months of tenure (47.4%).", body))

story.append(Paragraph(
"<b>Customer segments:</b> At Risk &ndash; 2,154 customers (avg. churn probability 0.75, avg. "
"monthly charges $76.20); Loyal &ndash; 4,626 customers (avg. churn probability 0.16); "
"Dormant &ndash; 263 new, short-tenure month-to-month customers (avg. churn probability 0.35).", body))

story.append(Paragraph("Conclusion", h2))
story.append(Paragraph(
"Contract type is by far the strongest predictor of churn, far outweighing demographic or usage "
"variables, indicating that contract-length incentives are the highest-leverage retention lever "
"available to the business. The Logistic Regression model achieved the best ROC-AUC (0.844), "
"reliably flagging at-risk customers so the retention team can prioritize outreach. Recommended "
"actions: (1) incentivize month-to-month customers to upgrade to annual contracts, (2) proactively "
"target the At-Risk segment with retention offers, (3) bundle online security/tech support into "
"fiber-optic plans, (4) encourage a shift away from electronic check payments, and (5) strengthen "
"onboarding in the first 90 days to reduce early-tenure churn.", body))

doc.build(story)
print("PDF report generated.")
