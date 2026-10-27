"""Dashboard layout as data: 4 pages plus 1 hidden drill-through page, canvas 1280 x 720 px.

Every visual has an exact position and field wells. tests/test_powerbi.py checks that visuals stay on the canvas, never
overlap, and only reference columns and measures that exist; src/powerbi_docs.py turns this into the spec and wireframes.
"""
from __future__ import annotations

from dataclasses import dataclass, field

CANVAS_W, CANVAS_H = 1280, 720
M = 16                                  # outer margin and gap between visuals


@dataclass
class Visual:
    id: str
    kind: str                           # text, slicer, card, bar, column, stacked_column, table, matrix
    title: str
    x: int
    y: int
    w: int
    h: int
    fields: dict[str, list[str]] = field(default_factory=dict)      # field wells; scanned for dangling references
    filters: list[str] = field(default_factory=list)                # visual-level filters, written as text
    notes: str = ""


@dataclass
class Page:
    name: str
    purpose: str
    visuals: list[Visual]
    hidden: bool = False
    interactions: str = ""


def _title(pid: str, text: str, subtitle: str) -> Visual:
    return Visual(f"{pid}_title", "text", text, M, 8, CANVAS_W - 2 * M, 40, {},
                  notes=f"Text box. Title in 20 pt semibold, subtitle in 11 pt grey: {subtitle}")


def _card(pid: str, measure: str, i: int, y: int = 112, w: int = 168, h: int = 96, gap: int = 12, notes: str = "") -> Visual:
    return Visual(f"{pid}_k{i}", "card", measure, M + i * (w + gap), y, w, h, {"values": [f"[{measure}]"]}, notes=notes)


def _churn_chart(pid: str, name: str, kind: str, title: str, axis: str, x: int, y: int, w: int, h: int, filters=None, notes="") -> Visual:
    return Visual(f"{pid}_{name}", kind, title, x, y, w, h,
                  {"axis": [axis], "values": ["[Churn Rate]"], "tooltips": ["[Total Customers]", "[Churned Customers]"],
                   "color_by_field_value": ["[Churn Bar Color]"], "constant_line": ["[Overall Churn Rate]"]},
                  filters=filters or [], notes="Axis starts at 0, data labels 0.0%, bars sorted by the dimension's order column. " + notes)


PAGES: list[Page] = [
    # ------------------------------------------------------------------------------------------ 1
    Page("Executive Overview",
         "Give management the churn picture in 10-15 seconds: how big, who, and how much money is exposed.",
         [_title("p1", "Customer churn: executive overview", "Risk metrics count active customers only; revenue means monthly charges (currency not stated)."),
          Visual("p1_s_contract", "slicer", "Contract", M, 56, 200, 48, {"field": ["DimContract[Contract]"]}, notes="Dropdown."),
          Visual("p1_s_internet", "slicer", "Internet service", 232, 56, 200, 48, {"field": ["DimInternetService[InternetService]"]}, notes="Dropdown."),
          Visual("p1_s_payment", "slicer", "Payment method", 448, 56, 200, 48, {"field": ["DimPaymentMethod[PaymentMethod]"]}, notes="Dropdown."),
          Visual("p1_s_tenure", "slicer", "Tenure group", 664, 56, 600, 48, {"field": ["DimTenureGroup[TenureGroup]"]}, notes="Horizontal tile slicer."),
          _card("p1", "Total Customers", 0), _card("p1", "Churned Customers", 1), _card("p1", "Churn Rate", 2),
          _card("p1", "Average Monthly Charges", 3), _card("p1", "Average Tenure (Months)", 4),
          _card("p1", "High Risk Customers", 5, notes="Active customers only. Label: 'High-risk customers (active)'."),
          _card("p1", "Revenue at Risk (High)", 6, notes="Monthly charges of active High-risk customers. Label: 'Revenue at risk (High, monthly)'."),
          _churn_chart("p1", "c_contract", "bar", "Churn rate by contract", "DimContract[Contract]", M, 224, 400, 400),
          _churn_chart("p1", "c_tenure", "column", "Churn rate by tenure group (months)", "DimTenureGroup[TenureGroup]", 432, 224, 400, 400,
                       notes="Stands in for a churn trend: the data has no dates, so tenure cohorts are the lifecycle view."),
          Visual("p1_c_risk", "column", "Active customers by risk tier", 848, 224, 416, 400,
                 {"axis": ["DimRiskCategory[RiskCategory]"], "values": ["[Active Customers]"], "tooltips": ["[Active Monthly Charges]", "[Expected Revenue at Risk]"],
                  "color_by_field_value": ["[Risk Color]"]},
                 notes="Columns left to right: Low, Medium, High (sorted by RiskOrder). Data labels on."),
          Visual("p1_note", "text", "How to read this page", M, 640, 1248, 64, {},
                 notes="Text box: 'Churn rate is the share of customers who have left. Risk tiers use predicted probability (High above 0.70, Medium 0.30-0.70, Low below 0.30). "
                       "Probabilities are a propensity from similar customers, not a forecast with a time horizon.'")],
         interactions="Slicers filter every visual and card. Clicking a bar cross-filters the cards and the other charts."),
    # ------------------------------------------------------------------------------------------ 2
    Page("Churn Drivers",
         "Show which customer characteristics go with higher churn, without claiming cause.",
         [_title("p2", "Churn drivers", "Associations in a single snapshot, not causes. Red bars are more than 2 points above the overall churn rate."),
          Visual("p2_s_senior", "slicer", "Senior citizen", M, 56, 300, 48, {"field": ["FactCustomer[SeniorCitizen]"]}, notes="Tile slicer (Yes / No)."),
          Visual("p2_s_partner", "slicer", "Partner", 332, 56, 300, 48, {"field": ["FactCustomer[Partner]"]}, notes="Tile slicer."),
          Visual("p2_s_dependents", "slicer", "Dependents", 648, 56, 300, 48, {"field": ["FactCustomer[Dependents]"]}, notes="Tile slicer."),
          _churn_chart("p2", "c_contract", "bar", "Churn rate by contract", "DimContract[Contract]", M, 112, 400, 288),
          _churn_chart("p2", "c_tenure", "column", "Churn rate by tenure group (months)", "DimTenureGroup[TenureGroup]", 432, 112, 400, 288),
          _churn_chart("p2", "c_internet", "bar", "Churn rate by internet service", "DimInternetService[InternetService]", 848, 112, 416, 288),
          _churn_chart("p2", "c_payment", "bar", "Churn rate by payment method", "DimPaymentMethod[PaymentMethod]", M, 416, 400, 288),
          _churn_chart("p2", "c_charge", "column", "Churn rate by monthly charge band", "DimChargeBand[ChargeBand]", 432, 416, 400, 288,
                       notes="Charge band is entangled with service type: fiber customers dominate the upper bands."),
          _churn_chart("p2", "c_addons", "column", "Churn rate by number of protective add-ons (internet customers)", "FactCustomer[ProtectiveAddOnCount]", 848, 416, 416, 288,
                       filters=["DimInternetService[InternetService] is not No"],
                       notes="Counts OnlineSecurity, OnlineBackup, DeviceProtection and TechSupport (0-4).")],
         interactions="Demographic slicers filter all six charts. Charts do not cross-filter each other (set Edit interactions to None) so every chart keeps its full comparison."),
    # ------------------------------------------------------------------------------------------ 3
    Page("Customer Risk",
         "Show which customers are likely to churn, and give analysts a customer-level list.",
         [_title("p3", "Customer risk", "Active customers, ranked by predicted churn probability. High-risk rows are shaded."),
          Visual("p3_s_contract", "slicer", "Contract", M, 56, 240, 48, {"field": ["DimContract[Contract]"]}, notes="Dropdown."),
          Visual("p3_s_tenure", "slicer", "Tenure group", 272, 56, 240, 48, {"field": ["DimTenureGroup[TenureGroup]"]}, notes="Dropdown."),
          Visual("p3_s_risk", "slicer", "Risk tier", 528, 56, 240, 48, {"field": ["DimRiskCategory[RiskCategory]"]}, notes="Dropdown."),
          Visual("p3_c_risk", "column", "Risk distribution (active customers)", M, 112, 300, 256,
                 {"axis": ["DimRiskCategory[RiskCategory]"], "values": ["[Active Customers]"], "tooltips": ["[Average Churn Probability (Active)]", "[Active Monthly Charges]"],
                  "color_by_field_value": ["[Risk Color]"]}, notes="Data labels on."),
          Visual("p3_c_contract", "stacked_column", "At-risk active customers by contract", 332, 112, 300, 256,
                 {"axis": ["DimContract[Contract]"], "legend": ["DimRiskCategory[RiskCategory]"], "values": ["[Active Customers]"]},
                 filters=["DimRiskCategory[RiskCategory] is Medium or High"], notes="Legend colours: Medium light red, High red."),
          Visual("p3_c_tenure", "stacked_column", "At-risk active customers by tenure group", 648, 112, 300, 256,
                 {"axis": ["DimTenureGroup[TenureGroup]"], "legend": ["DimRiskCategory[RiskCategory]"], "values": ["[Active Customers]"]},
                 filters=["DimRiskCategory[RiskCategory] is Medium or High"], notes="Same colours as the contract chart."),
          Visual("p3_c_hist", "column", "Distribution of churn probability (active customers)", 964, 112, 300, 256,
                 {"axis": ["FactCustomer[ProbabilityBin]"], "values": ["[Active Customers]"], "sort_by": ["FactCustomer[ProbabilityBinStart]"],
                  "color_rules_on": ["FactCustomer[ProbabilityBinStart]"]},
                 notes="Bin start below 0.30 grey-blue, 0.30 to 0.70 light red, 0.70 and above red (format by rules)."),
          Visual("p3_t_customers", "table", "Customer risk list", M, 384, 1248, 320,
                 {"columns": ["FactCustomer[CustomerID]", "DimContract[Contract]", "FactCustomer[tenure]", "FactCustomer[MonthlyCharges]", "FactCustomer[TotalCharges]",
                              "FactCustomer[ChurnProbability]", "DimRiskCategory[RiskCategory]", "FactCustomer[ExpectedMonthlyRevenueAtRisk]", "FactCustomer[TopRiskDriver1]"],
                  "color_by_field_value": ["[Risk Color]"]},
                 filters=["DimCustomerStatus[CustomerStatus] is Active"],
                 notes="Sort by ChurnProbability descending. ChurnProbability: data bars. RiskCategory: background colour from [Risk Color]. "
                       "Right-click a row, Drill through, Customer Detail.")],
         interactions="Slicers filter every visual on the page. The customer table is the drill-through source for the Customer Detail page."),
    # ------------------------------------------------------------------------------------------ 4
    Page("Retention Strategy",
         "Turn the analysis into prioritised, testable actions by risk and value.",
         [_title("p4", "Retention strategy", "Actions are hypotheses to validate with a pilot, not guaranteed fixes. Active customers only."),
          Visual("p4_k0", "card", "At-Risk Customers", M, 56, 300, 88, {"values": ["[At-Risk Customers]"]}, notes="Medium plus High risk, active."),
          Visual("p4_k1", "card", "Revenue at Risk (High)", 332, 56, 300, 88, {"values": ["[Revenue at Risk (High)]"]}, notes="Monthly charges of active High-risk customers."),
          Visual("p4_k2", "card", "Expected Revenue at Risk", 648, 56, 300, 88, {"values": ["[Expected Revenue at Risk]"]}, notes="Probability-weighted monthly exposure, not a forecast."),
          Visual("p4_k3", "card", "High-Value At-Risk Customers", 964, 56, 300, 88, {"values": ["[High-Value At-Risk Customers]"]}, notes="Top-quartile monthly charge, Medium or High risk."),
          Visual("p4_m_matrix", "matrix", "Active customers: risk tier by value segment", M, 160, 400, 272,
                 {"rows": ["DimRiskCategory[RiskCategory]"], "columns": ["FactCustomer[ValueSegment]"], "values": ["[Active Customers]"]},
                 notes="Background colour scale on the values (white to red). Rows sorted High, Medium, Low."),
          Visual("p4_c_exposure", "bar", "Expected revenue at risk by segment", 432, 160, 400, 272,
                 {"axis": ["DimRetentionSegment[RetentionSegment]"], "values": ["[Expected Revenue at Risk]"], "tooltips": ["[Active Customers]", "[Active Monthly Charges]"]},
                 filters=["DimRetentionSegment[RetentionSegment] is not Already churned"],
                 notes="Sorted by SegmentOrder. Volume matters: a large Medium segment can carry more exposure than a small High one."),
          Visual("p4_c_drivers", "bar", "Main model-associated risk driver (at-risk active customers)", 848, 160, 416, 272,
                 {"axis": ["FactCustomer[TopRiskDriverFeature]"], "values": ["[At-Risk Customers]"]},
                 notes="Each customer is counted under the feature that pushes their probability up the most. Model associations, not causes."),
          Visual("p4_t_segments", "table", "Segments and hypotheses to test", M, 448, 760, 256,
                 {"columns": ["DimRetentionSegment[RetentionSegment]", "[Active Customers]", "[Active Monthly Charges]", "[Average Churn Probability (Active)]",
                              "[Expected Revenue at Risk]", "DimRetentionSegment[SuggestedAction]"]},
                 filters=["DimRetentionSegment[RetentionSegment] is not Already churned"],
                 notes="SuggestedAction wraps text. Every action begins 'Hypothesis:' on purpose."),
          Visual("p4_t_priority", "table", "Top 10 customers to contact first", 792, 448, 472, 256,
                 {"columns": ["FactCustomer[CustomerID]", "DimRetentionSegment[RetentionSegment]", "[Expected Revenue at Risk]", "FactCustomer[TopRiskDriver1]"]},
                 filters=["Top N: 10 by [Expected Revenue at Risk]", "DimCustomerStatus[CustomerStatus] is Active"],
                 notes="Ranks by probability x monthly charge, so likelihood and value both count.")],
         interactions="Selecting a segment row filters the priority list. Cards respond to the matrix."),
    # ------------------------------------------------------------------------------------------ 5 (hidden)
    Page("Customer Detail",
         "Drill-through page: everything the model says about one customer.",
         [_title("p5", "Customer detail", "Reached by right-click, Drill through, from the Customer Risk table. Hidden from the page tabs."),
          Visual("p5_k0", "card", "Customer", M, 56, 300, 96, {"values": ["FactCustomer[CustomerID]"]}, notes="Aggregation: First."),
          Visual("p5_k1", "card", "Churn probability", 332, 56, 300, 96, {"values": ["[Average Predicted Probability]"]}, notes="One customer, so the average is the value."),
          Visual("p5_k2", "card", "Risk tier", 648, 56, 300, 96, {"values": ["DimRiskCategory[RiskCategory]"]}, notes="Aggregation: First."),
          Visual("p5_k3", "card", "Expected revenue at risk", 964, 56, 300, 96, {"values": ["[Expected Revenue at Risk]"]}),
          Visual("p5_t_drivers", "table", "What the model associates with this customer's risk", M, 168, 612, 200,
                 {"columns": ["FactCustomer[TopRiskDriver1]", "FactCustomer[TopRiskDriver2]", "FactCustomer[TopRiskDriver3]", "FactCustomer[TopProtectiveDriver1]"]},
                 notes="Associations from tree SHAP, not causes."),
          Visual("p5_t_profile", "table", "Customer profile", 644, 168, 620, 200,
                 {"columns": ["DimContract[Contract]", "DimInternetService[InternetService]", "DimPaymentMethod[PaymentMethod]", "FactCustomer[tenure]",
                              "FactCustomer[NumberOfServices]", "FactCustomer[MonthlyCharges]", "FactCustomer[ValueSegment]"]})],
         hidden=True,
         interactions="Drill-through field: FactCustomer[CustomerID]. Keep all filters off."),
]


def all_visuals() -> list[tuple[str, Visual]]:
    return [(p.name, v) for p in PAGES for v in p.visuals]
