# Insights Report: Loan Approval Dataset

## Scope and caveats
- Analysis covers 614 applications after cleaning.
- Descriptive analysis only. Associations are not causal, and the data's origin is undocumented.
- Gender, marital status, and property area are sensitive attributes. Differences across these groups are not evidence about how applicants should be treated.

## Key findings
1. **Approval rate is 68.7%** (Figure 1). The target is reasonably balanced.
2. **Credit history is associated with approval** (Figure 4):
   - Bad (0): 7.9% approved (n=89)
   - Good (1): 79.6% approved (n=475)
   - Unknown: 74.0% approved (n=50)
   - Gap between highest and lowest group: 71.7%.
3. **Property area differences** (Figure 5), gap between highest and lowest: 15.4%:
   - Rural: 61.5% approved (n=179)
   - Semiurban: 76.8% approved (n=233)
   - Urban: 65.8% approved (n=202)
   - Comparison: Credit history shows the larger gap, which is consistent with it being the stronger signal here.
4. **Income separates approvals only partly** (Figure 7). Median ApplicantIncome: approved 3,812, rejected 3,834. Check the scatter plot for overlap before claiming separation.
5. **Strongest numeric correlation** (Figure 6): ApplicantIncome and LoanAmount at r = 0.57.
6. **Median LoanAmount by area** (Figure 3): Rural 132, Semiurban 128, Urban 121 (thousands).

## Sensitivity to missing Credit_History
- Rows kept if dropped: 564 of 614 raw rows. Known-history approval rate: 68.3%, versus 68.7% with 'Unknown' kept.
- If these two numbers differ meaningfully, the Credit_History conclusion depends on how missing values were handled. Say so in the final report.

## Cleaning log
```
Raw shape: 614 rows x 13 columns
Missing values before cleaning:
Gender              13
Married              3
Dependents          15
Self_Employed       32
LoanAmount          22
Loan_Amount_Term    14
Credit_History      50
Exact duplicate rows: 0
Duplicate Loan_IDs: 0
Sensitivity check on Credit_History missing values:
  Rows kept if dropped: 564 of 614 (91.9%)
  Approval rate, all rows (Unknown kept as own group): 68.7%
  Approval rate, rows with known history only:          68.3%
  Approval rate, missing filled with mode (1):       68.7%
Removed 0 exact duplicate rows and 0 duplicate Loan_IDs.
Imputation applied:
  Gender: 13 missing -> mode
  Married: 3 missing -> mode
  Self_Employed: 32 missing -> mode
  Dependents: 15 missing -> mode
  LoanAmount: 22 missing -> median
  Loan_Amount_Term: 14 missing -> median
  Credit_History: 50 missing -> 'Unknown' category
Outliers (1.5 x IQR rule), flagged and kept:
  ApplicantIncome: 50 flagged
  CoapplicantIncome: 18 flagged
  LoanAmount: 41 flagged
Rows after cleaning: 614 (started with 614)
```

## Figures
- Figure 1: `figures\fig01_loan_status.png`
- Figure 2: `figures\fig02_income_raw_vs_log.png`
- Figure 3: `figures\fig03_loanamount_by_area.png`
- Figure 4: `figures\fig04_approval_by_credit_history.png`
- Figure 5: `figures\fig05_approval_by_property_area.png`
- Figure 6: `figures\fig06_correlation_heatmap.png`
- Figure 7: `figures\fig07_income_vs_loan_by_status.png`
