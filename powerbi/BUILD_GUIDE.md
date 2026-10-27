# Power BI build guide

This folder contains everything needed to build the churn dashboard in Power BI Desktop **except the `.pbix` file itself**.

## What is here, and what is not

| File | What it is | How it was made |
|---|---|---|
| `power_query.m` | Power Query (M) for the fact table and 8 dimensions | Generated from `src/powerbi_model.py` |
| `dax_measures.dax` | 27 documented DAX measures | Generated from `src/powerbi_model.py` |
| `data_model.md` | Star schema, relationships, sort-by and hide lists | Generated |
| `dashboard_spec.md` | Every visual on every page: type, position, fields, filters | Generated from `src/powerbi_pages.py` |
| `wireframes/` | One layout diagram per page | Generated from the same spec |
| `expected_values.md` | The numbers your dashboard should show | Pandas replica, cross-checked against the SQL results |
| `theme.json` | Colours and fonts | Hand-written |
| `customer_churn_dashboard.pbix` | **Not included** | See below |

**Why there is no `.pbix`.** Power BI Desktop is a Windows application, and a `.pbix` is a proprietary packaged format. It cannot be created reliably from Python or on the machine that built this project. A faked or hand-assembled file would risk looking finished while being broken, so the project gives you a precise specification instead. Expect roughly two to three hours the first time.

**What has been verified, and what has not.** The model, the measures and the layouts are defined once in Python, and `tests/test_powerbi.py` checks what can be checked without Power BI: that the star schema has no orphaned keys, that every column and measure referenced anywhere exists, that no visual overlaps another, that the measure values agree with the independent SQL results, and that these files are exactly what the generator produces. **Not verified:** that the M and DAX text parses in Power BI Desktop, and how the visuals look. Step 3 and Step 9 below are how you confirm that. If something does not work, treat it as a bug to fix in `src/powerbi_model.py`, then regenerate.

## 1. How the Python output reaches Power BI

```
raw CSV -> src.preprocessing -> churn_clean.csv -> src.train_model -> churn_model.pkl
                                                                         |
                                       src.predict (out-of-fold scoring) |
                                                                         v
                                     output/customer_churn_predictions.csv  --->  Power BI (Power Query)
```

Power BI never sees the model. It reads one file, `output/customer_churn_predictions.csv`, one row per customer with the probability, risk tier, value segment, drivers and revenue fields already computed. That is deliberate: the modelling is done and tested in Python, and Power BI does what it is good at, which is slicing, aggregating and presenting.

To refresh after the data or model changes: run `python -m src.predict`, then **Home > Refresh** in Power BI. The file location is held in a Power Query parameter (`DataPath`), so you change the path in one place.

## 2. Prerequisites

1. Power BI Desktop (free, Windows). On a Mac, use a Windows virtual machine or a cloud Windows desktop.
2. The project run through `python -m src.predict`, so `output/customer_churn_predictions.csv` exists (7,043 rows, 41 columns).

## 3. Load the data (Power Query)

**What Power Query is and why it is used:** it is the step that loads and shapes data before the model sees it. Doing the shaping here (types, extra columns, dimension tables) means it is repeatable on every refresh.

1. **Home > Transform data > Manage parameters > New parameter.** Name `DataPath`, Type Text, Current value = the full path to `customer_churn_predictions.csv`.
2. For each query in `power_query.m`: **New source > Blank query**, open **Advanced editor**, paste that section's code, and rename the query to the name in the section heading. Do `Customers_Raw` first. Right-click `Customers_Raw` and untick **Enable load** (it is only a staging step).
3. **Close & apply.**

**Check before moving on (this is the test the M text could not get here):**

- Every query loads without a yellow error bar.
- `FactCustomer` has **7,043 rows and 41 columns**. `CustomerID` shows 7,043 distinct values.
- Row counts of the dimensions: `DimContract` 3, `DimInternetService` 3, `DimPaymentMethod` 4, `DimTenureGroup` 4, `DimChargeBand` 4, `DimRiskCategory` 3, `DimCustomerStatus` 2, `DimRetentionSegment` 7.
- `ChurnProbability`, `MonthlyCharges`, `TotalCharges` are Decimal numbers, not text.

If a query errors, the usual causes are listed in section 9.

## 4. Build the model

Open **Model view** and follow `data_model.md`:

1. **Relationships.** Power BI may auto-create some. Make sure all eight exist, each many-to-one from the dimension to `FactCustomer`, single direction, active. (Table, columns and exact pairs are in `data_model.md`.)
2. **Why a star schema.** The fact table holds one row per customer and the numbers. The small dimension tables hold the labels and the correct order. Slicers and axes use the dimension, so one selection filters everything consistently, and sorting is defined once.
3. **Sort by column.** For each pair in `data_model.md` (for example `DimRiskCategory[RiskCategory]` by `RiskOrder`). Without this, `ChargeBand` reads 35-65, 65-95, <=35, >95.
4. **Hide** the columns listed in `data_model.md` (the foreign keys on the fact, the order and colour columns on the dimensions). Visuals use the dimension's column, which is why the foreign keys can be hidden.
5. **Summarization.** For numeric columns you do not want summed (`tenure`, `ChurnProbability`, `NumberOfServices`, `ProtectiveAddOnCount`), set Summarize by to **Don't summarize** or **Average** as appropriate. Measures do the real aggregation.

## 5. Create the measures

1. **Home > Enter data**, create a table named `_Measures` with one blank column, load it, then hide the column. This gives the measures a tidy home.
2. For each definition in `dax_measures.dax`: select `_Measures`, **Modeling > New measure**, paste it, then set the **Format** and **Display folder** printed above it.
3. **Why these are measures and not columns:** a measure is recalculated for whatever the user has selected, so one definition answers every slice. A calculated column would be fixed per row.

Three ideas worth understanding before you start, because they explain most of the DAX:

- **`CALCULATE` replaces filters.** `High Risk Customers` is `CALCULATE([Active Customers], DimRiskCategory[RiskCategory] = "High")`. It overrides whatever risk selection is on screen, so the measure always answers "how many active customers are High risk". The same logic means a Risk slicer will not change that card.
- **Active customers only.** Churned customers cannot be retained, so every risk and revenue-at-risk measure is built on `[Active Customers]` and `[Active Monthly Charges]`. The rule is written once, so no measure can forget it.
- **`DIVIDE` instead of `/`.** It returns blank for an empty selection instead of an error or a misleading 0%.

**Check:** add a Table visual with a few measures and compare with `expected_values.md`, section 2, first column ("No filter"). For example `[Total Customers]` is 7,043, `[Churn Rate]` is 26.5%, `[High Risk Customers]` is 65, `[Revenue at Risk (High)]` is 5,318, and `[Unmapped Customers]` is 0.

## 6. Apply the theme

**View > Themes > Browse for themes > `theme.json`.** It sets the palette (slate, dark slate, grey-blue, and one red `#C0504D` kept for risk) and Segoe UI fonts. Colour rules for individual visuals are in `dashboard_spec.md`.

## 7. Build the pages

1. For each page: **Format page > Canvas settings > Type: Custom, 1280 x 720 px**. White background.
2. Use `dashboard_spec.md` page by page, and `wireframes/` as the picture of the layout. Every visual lists its type, **x, y, width, height** (enter them under Format visual > General > Properties > Size and Position), its fields and any visual filter.
3. **Efficient order:** slicers, then the KPI cards, then the churn chart. The churn charts are all the same pattern (a dimension on the axis, `[Churn Rate]` as the value, colour by `[Churn Bar Color]`, a constant line at `[Overall Churn Rate]`), so build one, then copy and paste it and swap the axis field.
4. **Titles and labels:** descriptive titles as given in the spec, data labels on, gridlines off, axes starting at 0.
5. **Interactions:** set them as written in each page's "Interactions" line (Format > Edit interactions).
6. **Page 5 (Customer Detail)** is hidden and reached by drill-through; steps are at the end of `dashboard_spec.md`.

## 8. Conditional formatting

The recipes (bar colours, risk colours, data bars, the histogram's colour rules, the matrix gradient, the reference line) are in `dashboard_spec.md` under "Conditional formatting recipes". All risk colours come from one place: `DimRiskCategory[ColorHex]`, read by `[Risk Color]`. Change a colour there and every risk visual follows.

## 9. Check your build

Use `expected_values.md`, section 3, which lists what each key visual should show:

- Page 1: select **Contract = Month-to-month**. The cards should read 3,875 customers, 1,655 churned, 42.7% churn (section 2, second column).
- Page 1: the risk-tier chart should show 3,941 Low, 1,168 Medium and 65 High active customers.
- Page 2: the add-on chart should fall from 56.7% (none) to 5.3% (all four).
- Page 4: the segment table and the main-driver chart match the tables in section 3.

A number that does not match means a build error (a relationship, a filter, a column type), not a different truth.

**If something breaks**

| Symptom | Likely cause |
|---|---|
| `DataSource.NotFound` | `DataPath` is wrong. Edit the parameter. |
| Numbers load as text, or commas/decimals look swapped | A regional setting. The M types columns with the `en-US` culture; confirm the type step ran. |
| A slicer shows `(Blank)` | A fact value is missing from a dimension. `[Unmapped Customers]` will be above 0. Add the level to the dimension in `power_query.m`. |
| A measure shows 0 for High risk | A visual filter on churn status or risk tier is overriding it. Remove it from that visual. |
| Bars in the wrong order | The sort-by column was not set (step 4.3). |
| A chart shows every dimension level as `(Blank)` | The relationship is missing or inactive. |

## 10. Save and publish

1. Save as `powerbi/customer_churn_dashboard.pbix`.
2. Screenshot each of the four pages into `images/`: `overview.png`, `churn_drivers.png`, `customer_risk.png`, `retention_strategy.png`. These are the images the README links to. The wireframes in `powerbi/wireframes/` are layout diagrams, not screenshots.
3. **Optional, recommended for GitHub:** also **File > Export > Power BI template (.pbit)**. A template contains no data, so it is small and avoids redistributing the dataset; whoever opens it is prompted for `DataPath`. The dataset's own licence terms were not independently confirmed (see the README), which is another reason to prefer the template.

## 11. Reading the dashboard honestly

Put these on the dashboard (a text box on page 1) or in the README:

- **Probabilities are a propensity, not a forecast.** The data is a single snapshot with no time horizon. A customer's probability means "of customers like this one, this share have left". "Expected revenue at risk" is propensity-weighted exposure.
- **High risk is small among active customers.** Most high-probability customers have already churned, so only 65 of 5,174 active customers are High risk. The large exposure sits in the Medium tier, which is why page 4 shows value and segment, not only the High tier.
- **Revenue means monthly charges.** There is no margin, lifetime value, or currency in the data.
- **Everything is an association.** Drivers show what the model leans on, not what causes churn. The actions on page 4 are hypotheses to test with a pilot.
