"""
Team Standings Dashboard
A Dash web app for exploring team ranking data.

Each chart tab (bar, horizontal bar, pie, scatter, line, heatmap, correlation)
is rendered side-by-side with a live analytics panel containing:
  - Descriptive Analyst snapshot (summary stats + auto insights)
  - Forecasting Analyst snapshot (trend extrapolation + confidence band)
for the metric selected in the "Analytics metric" control.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, dash_table

# ----------------------------------------------------------------------------
# Data loading & enrichment
# ----------------------------------------------------------------------------
# The source CSV only has two columns (Standing, Team). To make richer chart
# types (pie, scatter, heatmap, correlation) and the analytics panel
# meaningful, we enrich the data with:
#   1. Confederation - a real-world factual mapping (which football
#      confederation each national team belongs to).
#   2. Simulated performance metrics (Points, Wins, Draws, Losses,
#      GoalsFor, GoalsAgainst, GoalDifference) - these are NOT real match
#      data, they are deterministically generated (seeded) so that better
#      standings tend to have better stats, purely to give the extra chart
#      and analytics features something numeric to work with. This is
#      clearly labeled in the UI as simulated/demo data.

DATA_PATH = "data/team_standings.csv"

CONFEDERATION_MAP = {
    "Spain": "UEFA", "Netherlands": "UEFA", "Germany": "UEFA", "Uruguay": "CONMEBOL",
    "Argentina": "CONMEBOL", "Brazil": "CONMEBOL", "Ghana": "CAF", "Paraguay": "CONMEBOL",
    "Japan": "AFC", "Chile": "CONMEBOL", "Portugal": "UEFA", "USA": "CONCACAF",
    "England": "UEFA", "Mexico": "CONCACAF", "South Korea": "AFC", "Slovakia": "UEFA",
    "Ivory Coast": "CAF", "Slovenia": "UEFA", "Switzerland": "UEFA", "South Africa": "CAF",
    "Australia": "AFC", "New Zealand": "OFC", "Serbia": "UEFA", "Denmark": "UEFA",
    "Greece": "UEFA", "Italy": "UEFA", "Nigeria": "CAF", "Algeria": "CAF",
    "France": "UEFA", "Honduras": "CONCACAF", "Cameroon": "CAF", "North Korea": "AFC",
}

NUMERIC_COLS = [
    "Points", "Wins", "Draws", "Losses", "GoalsFor", "GoalsAgainst", "GoalDifference",
]


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.sort_values("Standing").reset_index(drop=True)
    return df


def enrich_data(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    df = df.copy()
    df["Confederation"] = df["Team"].map(CONFEDERATION_MAP).fillna("Other")

    rng = np.random.default_rng(seed)
    n = len(df)
    max_standing = df["Standing"].max()

    # Better (lower) standing -> higher skill score, with noise.
    skill = (max_standing - df["Standing"] + 1) / max_standing
    noise = rng.normal(0, 0.08, size=n)
    skill = np.clip(skill + noise, 0.05, 1.0)

    games_played = 4  # e.g. group-stage style
    wins = np.round(skill * games_played * rng.uniform(0.5, 1.0, size=n)).astype(int)
    wins = np.clip(wins, 0, games_played)
    remaining = games_played - wins
    draws = np.array([rng.integers(0, r + 1) if r > 0 else 0 for r in remaining])
    losses = games_played - wins - draws

    goals_for = np.round(wins * rng.uniform(1.2, 2.2, size=n) + draws * rng.uniform(0.5, 1.2, size=n)).astype(int)
    goals_against = np.round(losses * rng.uniform(1.0, 2.0, size=n) + draws * rng.uniform(0.3, 1.0, size=n)).astype(int)
    points = wins * 3 + draws

    df["Wins"] = wins
    df["Draws"] = draws
    df["Losses"] = losses
    df["GoalsFor"] = goals_for
    df["GoalsAgainst"] = goals_against
    df["GoalDifference"] = goals_for - goals_against
    df["Points"] = points

    return df


df = enrich_data(load_data())
TOTAL_TEAMS = len(df)
CONFEDERATIONS = sorted(df["Confederation"].unique())
METRIC_OPTIONS = ["Standing"] + NUMERIC_COLS

# ----------------------------------------------------------------------------
# App setup
# ----------------------------------------------------------------------------

app = Dash(__name__, title="Team Standings Dashboard")
server = app.server  # exposed for WSGI deployment (gunicorn etc.)

CARD_STYLE = {
    "backgroundColor": "#ffffff",
    "borderRadius": "12px",
    "padding": "20px",
    "boxShadow": "0 2px 8px rgba(0,0,0,0.08)",
}

PANEL_STYLE = {
    "backgroundColor": "#fbfcfd",
    "borderRadius": "10px",
    "padding": "16px",
    "border": "1px solid #e6e9ee",
}

PAGE_STYLE = {
    "backgroundColor": "#f4f6f9",
    "minHeight": "100vh",
    "fontFamily": "'Segoe UI', Arial, sans-serif",
    "padding": "30px 40px",
}

BASE_LAYOUT = dict(plot_bgcolor="white", paper_bgcolor="white", margin=dict(t=50, b=30, l=30, r=20))

# ----------------------------------------------------------------------------
# Layout
# ----------------------------------------------------------------------------

app.layout = html.Div(
    style=PAGE_STYLE,
    children=[
        html.Div(
            [
                html.H1("🏆 Team Standings Dashboard", style={"marginBottom": "4px"}),
                html.P(
                    f"Exploring {TOTAL_TEAMS} teams ranked by standing.",
                    style={"color": "#666", "marginTop": 0},
                ),
                html.P(
                    "Note: Standing, Team, and Confederation are real. "
                    "Wins/Draws/Losses/Goals/Points are simulated demo metrics "
                    "(seeded so better standings trend toward better stats) so "
                    "the charts and the analytics panel have numeric data to work with.",
                    style={"color": "#999", "fontSize": "13px", "marginTop": "4px", "maxWidth": "800px"},
                ),
            ],
            style={"marginBottom": "24px"},
        ),

        # --- Summary cards ---
        html.Div(
            [
                html.Div(
                    [html.H3(str(TOTAL_TEAMS)), html.P("Total Teams")],
                    style={**CARD_STYLE, "flex": 1, "textAlign": "center"},
                ),
                html.Div(
                    [html.H3(df.iloc[0]["Team"]), html.P("#1 Ranked")],
                    style={**CARD_STYLE, "flex": 1, "textAlign": "center"},
                ),
                html.Div(
                    [html.H3(df.iloc[-1]["Team"]), html.P(f"#{TOTAL_TEAMS} Ranked")],
                    style={**CARD_STYLE, "flex": 1, "textAlign": "center"},
                ),
            ],
            style={"display": "flex", "gap": "16px", "marginBottom": "24px"},
        ),

        # --- Filter controls ---
        html.Div(
            [
                html.Div(
                    [
                        html.Label("Search team"),
                        dcc.Input(
                            id="search-input",
                            type="text",
                            placeholder="e.g. Brazil",
                            style={"width": "100%", "padding": "8px", "marginTop": "4px"},
                        ),
                    ],
                    style={"flex": 1},
                ),
                html.Div(
                    [
                        html.Label("Standing range"),
                        dcc.RangeSlider(
                            id="range-slider",
                            min=int(df["Standing"].min()),
                            max=int(df["Standing"].max()),
                            value=[int(df["Standing"].min()), int(df["Standing"].max())],
                            step=1,
                            marks={
                                i: str(i)
                                for i in range(
                                    int(df["Standing"].min()),
                                    int(df["Standing"].max()) + 1,
                                    5,
                                )
                            },
                        ),
                    ],
                    style={"flex": 2, "paddingLeft": "24px"},
                ),
                html.Div(
                    [
                        html.Label("Top N to chart"),
                        dcc.Dropdown(
                            id="top-n-dropdown",
                            options=[
                                {"label": f"Top {n}", "value": n}
                                for n in [5, 10, 15, 20, TOTAL_TEAMS]
                            ],
                            value=10,
                            clearable=False,
                        ),
                    ],
                    style={"flex": 1, "paddingLeft": "24px"},
                ),
            ],
            style={**CARD_STYLE, "display": "flex", "marginBottom": "24px"},
        ),

        # --- Analytics controls (drive the side panel next to every chart) ---
        html.Div(
            [
                html.Div(
                    [
                        html.Label("Analytics metric"),
                        dcc.Dropdown(
                            id="metric-dropdown",
                            options=[{"label": m, "value": m} for m in METRIC_OPTIONS],
                            value="Points",
                            clearable=False,
                        ),
                    ],
                    style={"flex": 1},
                ),
                html.Div(
                    [
                        html.Label("Forecast horizon (future ranks)"),
                        dcc.Slider(
                            id="forecast-horizon",
                            min=1, max=15, step=1, value=5,
                            marks={i: str(i) for i in range(1, 16, 2)},
                        ),
                    ],
                    style={"flex": 2, "paddingLeft": "24px"},
                ),
            ],
            style={**CARD_STYLE, "display": "flex", "marginBottom": "24px"},
        ),

        # --- Tabs + side-by-side chart / analytics panel ---
        html.Div(
            [
                dcc.Tabs(
                    id="chart-tabs",
                    value="tab-bar",
                    children=[
                        dcc.Tab(label="Bar", value="tab-bar"),
                        dcc.Tab(label="Horizontal Bar", value="tab-barh"),
                        dcc.Tab(label="Pie", value="tab-pie"),
                        dcc.Tab(label="Scatter", value="tab-scatter"),
                        dcc.Tab(label="Line", value="tab-line"),
                        dcc.Tab(label="Heatmap", value="tab-heatmap"),
                        dcc.Tab(label="Correlation", value="tab-correlation"),
                    ],
                ),
                html.Div(
                    [
                        html.Div(id="chart-panel", style={"flex": "3", "minWidth": 0}),
                        html.Div(id="analytics-panel", style={"flex": "2", "minWidth": 0, "marginLeft": "16px"}),
                    ],
                    style={"display": "flex", "marginTop": "16px", "alignItems": "flex-start"},
                ),
            ],
            style={**CARD_STYLE, "marginBottom": "24px"},
        ),

        # --- Table ---
        html.Div(
            dash_table.DataTable(
                id="standings-table",
                columns=[{"name": c, "id": c} for c in df.columns],
                data=df.to_dict("records"),
                page_size=12,
                sort_action="native",
                style_table={"overflowX": "auto"},
                style_cell={
                    "textAlign": "left",
                    "padding": "8px",
                    "fontFamily": "'Segoe UI', Arial, sans-serif",
                },
                style_header={
                    "backgroundColor": "#2c3e50",
                    "color": "white",
                    "fontWeight": "bold",
                },
                style_data_conditional=[
                    {"if": {"row_index": "odd"}, "backgroundColor": "#f9fafb"}
                ],
            ),
            style=CARD_STYLE,
        ),
    ],
)

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def filter_data(search_value, standing_range):
    filtered = df.copy()
    low, high = standing_range
    filtered = filtered[(filtered["Standing"] >= low) & (filtered["Standing"] <= high)]
    if search_value:
        filtered = filtered[
            filtered["Team"].str.contains(search_value, case=False, na=False)
        ]
    return filtered.sort_values("Standing")


def empty_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        xaxis={"visible": False},
        yaxis={"visible": False},
        annotations=[{"text": message, "showarrow": False, "font": {"size": 16}}],
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    return fig


def empty_message(message: str):
    return html.Div(message, style={"padding": "24px", "textAlign": "center", "color": "#888"})


# --- Chart builders (left panel) -----------------------------------------

def make_bar(chart_df, top_n):
    d = chart_df.head(top_n)
    fig = px.bar(
        d, x="Team", y="Standing",
        title=f"Standings — Bar (top {len(d)} of filtered results)",
        color="Standing", color_continuous_scale="Blues_r", text="Standing",
    )
    fig.update_yaxes(autorange="reversed", title="Standing (1 = best)")
    fig.update_layout(showlegend=False, **BASE_LAYOUT)
    return fig


def make_barh(chart_df, top_n):
    d = chart_df.head(top_n).sort_values("Standing", ascending=False)
    fig = px.bar(
        d, x="Points", y="Team", orientation="h",
        title=f"Points — Horizontal Bar (top {len(d)} of filtered results)",
        color="Points", color_continuous_scale="Greens", text="Points",
    )
    fig.update_layout(showlegend=False, **BASE_LAYOUT)
    return fig


def make_pie(chart_df):
    counts = chart_df["Confederation"].value_counts().reset_index()
    counts.columns = ["Confederation", "Count"]
    fig = px.pie(
        counts, names="Confederation", values="Count",
        title="Team Distribution by Confederation (filtered set)", hole=0.35,
    )
    fig.update_layout(**BASE_LAYOUT)
    return fig


def make_scatter(chart_df):
    fig = px.scatter(
        chart_df, x="Wins", y="GoalDifference", size="Points", color="Confederation",
        hover_name="Team", title="Wins vs Goal Difference (bubble size = Points)",
    )
    fig.update_layout(**BASE_LAYOUT)
    return fig


def make_line(chart_df):
    d = chart_df.sort_values("Standing")
    fig = px.line(
        d, x="Standing", y="Points", markers=True, hover_name="Team",
        title="Points Trend Across Standing Order",
    )
    fig.update_layout(**BASE_LAYOUT)
    return fig


def make_heatmap(chart_df):
    pivot = chart_df.groupby("Confederation")[NUMERIC_COLS].mean().round(1)
    fig = px.imshow(
        pivot, text_auto=True, aspect="auto", color_continuous_scale="YlOrRd",
        title="Average Performance Metrics by Confederation (Heatmap)",
        labels=dict(x="Metric", y="Confederation", color="Avg. value"),
    )
    fig.update_layout(**BASE_LAYOUT)
    return fig


def make_correlation(chart_df):
    corr_cols = ["Standing"] + NUMERIC_COLS
    corr = chart_df[corr_cols].corr().round(2)
    fig = px.imshow(
        corr, text_auto=True, aspect="auto", color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1, title="Correlation Matrix of Numeric Metrics",
    )
    fig.update_layout(**BASE_LAYOUT)
    return fig


def build_chart_panel(tab, chart_df, top_n):
    if chart_df.empty:
        return dcc.Graph(figure=empty_figure("No teams match the current filters."))
    if tab == "tab-bar":
        return dcc.Graph(figure=make_bar(chart_df, top_n))
    if tab == "tab-barh":
        return dcc.Graph(figure=make_barh(chart_df, top_n))
    if tab == "tab-pie":
        return dcc.Graph(figure=make_pie(chart_df))
    if tab == "tab-scatter":
        return dcc.Graph(figure=make_scatter(chart_df))
    if tab == "tab-line":
        return dcc.Graph(figure=make_line(chart_df))
    if tab == "tab-heatmap":
        return dcc.Graph(figure=make_heatmap(chart_df))
    if tab == "tab-correlation":
        if len(chart_df) < 3:
            return dcc.Graph(figure=empty_figure(
                "Need at least 3 teams in the filtered set for a correlation matrix."
            ))
        return dcc.Graph(figure=make_correlation(chart_df))
    return dcc.Graph(figure=empty_figure("Select a chart type."))


# --- Descriptive analyst snapshot (right panel, top half) -----------------

def build_descriptive_snapshot(chart_df, metric):
    if len(chart_df) < 2:
        return empty_message("Need at least 2 teams in the filtered set for descriptive stats.")

    mean_val = chart_df[metric].mean()
    std_val = chart_df[metric].std()
    skew_val = chart_df[metric].skew()
    corr_with_standing = chart_df[["Standing", metric]].corr().iloc[0, 1]

    conf_avg = chart_df.groupby("Confederation")[metric].mean().sort_values(ascending=False)
    top_conf = conf_avg.index[0] if len(conf_avg) else "N/A"
    bottom_conf = conf_avg.index[-1] if len(conf_avg) else "N/A"

    skew_desc = "roughly symmetric"
    if pd.notna(skew_val):
        if skew_val > 0.5:
            skew_desc = "right-skewed"
        elif skew_val < -0.5:
            skew_desc = "left-skewed"

    corr_desc = "little relationship with"
    if pd.notna(corr_with_standing):
        if abs(corr_with_standing) >= 0.7:
            corr_desc = "a strong relationship with"
        elif abs(corr_with_standing) >= 0.4:
            corr_desc = "a moderate relationship with"

    mini_hist = px.histogram(
        chart_df, x=metric, nbins=10, color_discrete_sequence=["#2c7fb8"],
    )
    mini_hist.update_layout(height=180, showlegend=False, **BASE_LAYOUT)
    mini_hist.update_layout(margin=dict(t=10, b=20, l=20, r=10))

    stats_row = html.Div(
        [
            html.Div([html.Div(f"{chart_df[metric].min():.1f}", style={"fontWeight": "bold"}), html.Div("Min", style={"fontSize": "11px", "color": "#888"})], style={"textAlign": "center", "flex": 1}),
            html.Div([html.Div(f"{mean_val:.1f}", style={"fontWeight": "bold"}), html.Div("Mean", style={"fontSize": "11px", "color": "#888"})], style={"textAlign": "center", "flex": 1}),
            html.Div([html.Div(f"{chart_df[metric].median():.1f}", style={"fontWeight": "bold"}), html.Div("Median", style={"fontSize": "11px", "color": "#888"})], style={"textAlign": "center", "flex": 1}),
            html.Div([html.Div(f"{std_val:.1f}", style={"fontWeight": "bold"}), html.Div("Std Dev", style={"fontSize": "11px", "color": "#888"})], style={"textAlign": "center", "flex": 1}),
            html.Div([html.Div(f"{chart_df[metric].max():.1f}", style={"fontWeight": "bold"}), html.Div("Max", style={"fontSize": "11px", "color": "#888"})], style={"textAlign": "center", "flex": 1}),
        ],
        style={"display": "flex", "marginBottom": "10px"},
    )

    insights = [
        f"{metric} is {skew_desc} across {len(chart_df)} team(s) (skew = {skew_val:.2f}).",
        f"{metric} shows {corr_desc} Standing (r = {corr_with_standing:.2f}).",
        f"'{top_conf}' leads on average {metric}; '{bottom_conf}' trails.",
    ]

    return html.Div([
        html.H4(f"📊 Descriptive Analyst — {metric}", style={"marginTop": 0, "marginBottom": "8px"}),
        stats_row,
        dcc.Graph(figure=mini_hist, config={"displayModeBar": False}),
        html.Ul([html.Li(i, style={"fontSize": "13px"}) for i in insights], style={"paddingLeft": "18px"}),
    ], style={**PANEL_STYLE, "marginBottom": "16px"})


# --- Forecasting analyst snapshot (right panel, bottom half) --------------

def linear_forecast(x, y, horizon):
    """Simple OLS linear trend fit + extrapolation with an approximate 95% CI band."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)

    slope, intercept = np.polyfit(x, y, 1)
    y_fit = slope * x + intercept
    residuals = y - y_fit
    dof = max(n - 2, 1)
    residual_std = np.sqrt(np.sum(residuals ** 2) / dof)

    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    future_x = np.arange(x.max() + 1, x.max() + 1 + horizon)
    future_y = slope * future_x + intercept

    x_mean = x.mean()
    x_var = np.sum((x - x_mean) ** 2) if n > 1 else 1.0
    se_future = residual_std * np.sqrt(1 + 1 / n + (future_x - x_mean) ** 2 / x_var)
    ci = 1.96 * se_future

    return {
        "slope": slope, "intercept": intercept, "r_squared": r_squared,
        "fitted_x": x, "fitted_y": y_fit,
        "future_x": future_x, "future_y": future_y,
        "future_lower": future_y - ci, "future_upper": future_y + ci,
    }


def build_forecast_snapshot(chart_df, metric, horizon):
    if len(chart_df) < 3:
        return empty_message("Need at least 3 teams in the filtered set to fit a forecast trend.")

    d = chart_df.sort_values("Standing")
    result = linear_forecast(d["Standing"], d[metric], horizon)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=d["Standing"], y=d[metric], mode="markers", name="Actual",
        marker=dict(color="#2c7fb8", size=7), text=d["Team"], hovertemplate="%{text}: %{y}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=result["fitted_x"], y=result["fitted_y"], mode="lines", name="Trend",
        line=dict(color="#e34a33", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=result["future_x"], y=result["future_y"], mode="lines+markers", name="Forecast",
        line=dict(color="#e34a33", width=2, dash="dash"),
    ))
    fig.add_trace(go.Scatter(
        x=np.concatenate([result["future_x"], result["future_x"][::-1]]),
        y=np.concatenate([result["future_upper"], result["future_lower"][::-1]]),
        fill="toself", fillcolor="rgba(227,74,51,0.15)", line=dict(color="rgba(0,0,0,0)"),
        name="95% CI", hoverinfo="skip",
    ))
    fig.update_layout(height=220, legend=dict(orientation="h", y=-0.25), **BASE_LAYOUT)
    fig.update_layout(margin=dict(t=10, b=30, l=30, r=10))

    trend_word = "increases" if result["slope"] > 0 else "decreases"
    insights = [
        f"{metric} {trend_word} ~{abs(result['slope']):.2f} per rank step (R² = {result['r_squared']:.2f}).",
        f"+{horizon} ranks beyond current range → projected {metric} ≈ {result['future_y'][-1]:.2f} "
        f"(95% CI: {result['future_lower'][-1]:.2f} to {result['future_upper'][-1]:.2f}).",
        "Linear trend extrapolation over rank order — illustrative, not a true time-series forecast.",
    ]

    return html.Div([
        html.H4(f"🔮 Forecasting Analyst — {metric}", style={"marginTop": 0, "marginBottom": "8px"}),
        dcc.Graph(figure=fig, config={"displayModeBar": False}),
        html.Ul([html.Li(i, style={"fontSize": "13px"}) for i in insights], style={"paddingLeft": "18px"}),
    ], style=PANEL_STYLE)


def build_analytics_panel(chart_df, metric, horizon):
    return html.Div([
        build_descriptive_snapshot(chart_df, metric),
        build_forecast_snapshot(chart_df, metric, horizon),
    ])


# ----------------------------------------------------------------------------
# Callbacks
# ----------------------------------------------------------------------------

@app.callback(
    Output("standings-table", "data"),
    Input("search-input", "value"),
    Input("range-slider", "value"),
)
def update_table(search_value, standing_range):
    filtered = filter_data(search_value, standing_range)
    return filtered.to_dict("records")


@app.callback(
    Output("chart-panel", "children"),
    Input("chart-tabs", "value"),
    Input("search-input", "value"),
    Input("range-slider", "value"),
    Input("top-n-dropdown", "value"),
)
def update_chart_panel(tab, search_value, standing_range, top_n):
    chart_df = filter_data(search_value, standing_range)
    return build_chart_panel(tab, chart_df, top_n)


@app.callback(
    Output("analytics-panel", "children"),
    Input("search-input", "value"),
    Input("range-slider", "value"),
    Input("metric-dropdown", "value"),
    Input("forecast-horizon", "value"),
)
def update_analytics_panel(search_value, standing_range, metric, horizon):
    chart_df = filter_data(search_value, standing_range)
    return build_analytics_panel(chart_df, metric, horizon)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)
