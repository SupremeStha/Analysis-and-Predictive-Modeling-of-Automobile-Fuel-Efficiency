# Analysis and Predictive Modeling of Automobile Fuel Efficiency

**What makes a car fuel-efficient — and can we predict it?**

This project digs into the classic Auto MPG dataset to answer that question, walking the full path from raw, messy data to a working predictive model. Along the way it tests real statistical hypotheses, compares two modeling approaches head-to-head, and finds natural groupings among vehicles using unsupervised learning.

---

## The Question

Fuel economy depends on a tangle of interacting factors — how big the engine is, how heavy the car is, how old the design is. This project asks three things:

1. Which vehicle characteristics actually predict MPG, and by how much?
2. Does a car's region of origin (USA, Japan, Europe) make a statistically significant difference to its fuel economy?
3. Do vehicles naturally cluster into distinct groups based on their specs?

## Dataset

The classic **Auto MPG** dataset — one vehicle per row, with:

| Feature | Description |
|---|---|
| `mpg` | Miles per gallon (target variable) |
| `cylinders` | Number of engine cylinders |
| `displacement` | Engine displacement |
| `horsepower` | Engine horsepower |
| `weight` | Vehicle weight |
| `acceleration` | 0–60 acceleration time |
| `model_year` | Year the model was released |
| `origin` | Region of manufacture (USA / Japan / Europe) |
| `name` | Make and model, later split into company and model name |

## Workflow

```
Raw Data → Cleaning → EDA → Hypothesis Testing → Outlier Handling → Modeling → Clustering
```

**1. Cleaning**
Vehicle names were split into company and model. Missing `horsepower` values (recorded as `?`) were converted and imputed with the median. Outliers were handled using winsorization rather than removal, to keep the dataset intact while limiting the influence of extreme values.

**2. Exploratory Analysis**
Distributions of every numeric feature were visualized, followed by a full correlation heatmap to see which variables move together with MPG before any modeling began.

**3. Hypothesis Testing**
A one-way ANOVA tested whether MPG genuinely differs across region of origin, followed by Tukey post-hoc tests to identify exactly which regional pairs differ significantly.

**4. Predictive Modeling**
Two models were trained and compared on identical train/test splits:

| Model | MSE | R² |
|---|---|---|
| Linear Regression | 9.58 | 0.84 |
| **Random Forest** | **5.60** | **0.91** |

The Random Forest model came out ahead on both metrics, explaining roughly 91% of the variance in MPG.

**5. Feature Importance**
The Random Forest model was interrogated to see which features it actually relied on:

| Rank | Feature | Importance |
|---|---|---|
| 1 | Displacement | 0.365 |
| 2 | Weight | 0.222 |
| 3 | Model Year | 0.132 |
| 4 | Cylinders | 0.130 |
| 5 | Horsepower | 0.123 |
| 6 | Acceleration | 0.022 |
| 7 | Origin | 0.006 |

Engine displacement and vehicle weight dominate the prediction — origin, despite the earlier hypothesis test, turns out to add almost nothing once the physical specs are accounted for.

**6. Clustering**
Beyond prediction, K-means clustering (with the optimal cluster count chosen via the elbow method) was used to group vehicles by shared characteristics like weight and MPG, revealing natural "classes" of vehicle in the data without using any labels.

## Tech Stack

`Python` · `Pandas` · `NumPy` · `Matplotlib` / `Seaborn` · `SciPy` · `Scikit-learn`

Specific techniques: Linear Regression, Random Forest Regression, K-Means Clustering, ANOVA, Tukey HSD, Winsorization, Label Encoding, Standard Scaling.

## Repository Contents

```
├── Main Code.ipynb     # Full analysis notebook
├── Automobile.csv       # Dataset
├── Report.pdf            # Written report of findings
└── README.md
```

## Key Takeaways

- A Random Forest model can predict a car's MPG to within roughly ±2.4 MPG on average (RMSE ≈ √5.60), noticeably outperforming a linear baseline.
- Engine displacement and vehicle weight are the two strongest physical predictors of fuel efficiency — no surprise to anyone who's pushed a heavy car up a hill, but satisfying to see confirmed in the data.
- Region of origin shows up as statistically significant in isolation, but loses almost all predictive power once the vehicle's actual physical specs are in the model — a good reminder that statistical significance and predictive importance aren't the same thing.

## Getting Started

```bash
git clone https://github.com/SupremeStha/Analysis-and-Predictive-Modeling-of-Automobile-Fuel-Efficiency.git
cd Analysis-and-Predictive-Modeling-of-Automobile-Fuel-Efficiency
pip install pandas numpy matplotlib seaborn scipy scikit-learn statsmodels
jupyter notebook "Main Code.ipynb"
```
