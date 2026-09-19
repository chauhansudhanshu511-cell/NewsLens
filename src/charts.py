"""Plotly figures used by the Streamlit app."""

from __future__ import annotations

import plotly.graph_objects as go

REAL_COLOR = "#2E6FD8"   # blue
FAKE_COLOR = "#E0762B"   # orange
GRID = "rgba(128,128,128,0.25)"


def _base_layout(fig: go.Figure, height: int) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=13),
    )
    return fig


def probability_bar(prob_real: float, prob_fake: float) -> go.Figure:
    """One horizontal bar split into the model's real vs fake estimate."""
    fig = go.Figure()
    for name, value, color in [("Real (model estimate)", prob_real, REAL_COLOR),
                               ("Fake (model estimate)", prob_fake, FAKE_COLOR)]:
        fig.add_trace(go.Bar(
            y=["Model estimate"], x=[value * 100], name=name, orientation="h",
            marker_color=color, text=f"{value:.0%}", textposition="inside",
            insidetextanchor="middle", hovertemplate=f"{name}: %{{x:.1f}}%<extra></extra>",
        ))
    fig.update_layout(barmode="stack", showlegend=True,
                      legend=dict(orientation="h", y=-0.35, x=0))
    fig.update_xaxes(range=[0, 100], ticksuffix="%", gridcolor=GRID)
    fig.update_yaxes(showticklabels=False)
    fig.add_vline(x=50, line_dash="dot", line_color="gray")
    return _base_layout(fig, 150)


def contributions_bar(terms: dict) -> go.Figure:
    """Horizontal bars: positive = pushed toward fake, negative = toward real."""
    rows = sorted(terms["toward_fake"] + terms["toward_real"], key=lambda r: r[1])
    names = [r[0] for r in rows]
    values = [r[1] for r in rows]
    colors = [FAKE_COLOR if v > 0 else REAL_COLOR for v in values]
    fig = go.Figure(go.Bar(x=values, y=names, orientation="h", marker_color=colors,
                           hovertemplate="%{y}: %{x:+.3f}<extra></extra>"))
    fig.update_xaxes(title="Contribution to log-odds  (← real | fake →)", gridcolor=GRID,
                     zeroline=True, zerolinecolor="gray")
    return _base_layout(fig, max(220, 26 * len(rows) + 60))


def confusion_matrix_fig(cm: list[list[int]], labels: list[str]) -> go.Figure:
    total = sum(sum(r) for r in cm) or 1
    text = [[f"{v:,}<br>({v / total:.1%})" for v in row] for row in cm]
    fig = go.Figure(go.Heatmap(
        z=cm, x=[f"Predicted {l}" for l in labels], y=[f"Actual {l}" for l in labels],
        text=text, texttemplate="%{text}", colorscale="Blues", showscale=False,
        hovertemplate="%{y} / %{x}: %{z:,}<extra></extra>",
    ))
    fig.update_yaxes(autorange="reversed")
    return _base_layout(fig, 300)
