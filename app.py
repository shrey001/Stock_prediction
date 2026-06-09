import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="AI Stock Picks",
    layout="wide"
)

st.title("AI Stock Picks")
st.caption("For education only — not financial advice.")

ranking = pd.read_csv("version6_current_ranking.csv")
portfolio = pd.read_csv("version6_diversified_portfolio.csv")
importance = pd.read_csv("version6_feature_importance.csv")

ranking["AI Score"] = (ranking["Probability_Top_20"] * 100).round(1)
portfolio["AI Score"] = (portfolio["Probability_Top_20"] * 100).round(1)

def simple_signal(score):
    if score >= 50:
        return "Strong pick"
    elif score >= 25:
        return "Worth watching"
    return "Lower priority"

def risk_level(volatility):
    try:
        v = float(volatility)
        if v >= 0.04:
            return "Higher risk"
        elif v >= 0.025:
            return "Medium risk"
        return "Lower risk"
    except:
        return "Unknown risk"

ranking["Simple View"] = ranking["AI Score"].apply(simple_signal)
ranking["Risk Level"] = ranking["Volatility_20d"].apply(risk_level)

portfolio["Simple View"] = portfolio["AI Score"].apply(simple_signal)
portfolio["Risk Level"] = portfolio["Volatility_20d"].apply(risk_level)

show_cols = [
    "Ticker",
    "Sector",
    "AI Score",
    "Simple View",
    "Risk Level",
    "Latest_Date"
]

tab1, tab2, tab3, tab4 = st.tabs([
    "Top picks",
    "Stock detail",
    "Suggested portfolio",
    "What the AI is using"
])

with tab1:
    st.subheader("Top stocks right now")

    sector_options = ["All"] + sorted(ranking["Sector"].dropna().unique().tolist())
    sector_filter = st.selectbox("Filter by business category", sector_options)

    view_options = ["All", "Strong pick", "Worth watching", "Lower priority"]
    view_filter = st.selectbox("Filter by AI opinion", view_options)

    filtered = ranking.copy()

    if sector_filter != "All":
        filtered = filtered[filtered["Sector"] == sector_filter]

    if view_filter != "All":
        filtered = filtered[filtered["Simple View"] == view_filter]

    st.dataframe(
        filtered[show_cols].head(50),
        use_container_width=True,
        hide_index=True
    )

with tab2:
    st.subheader("Drill down into one stock")

    selected_ticker = st.selectbox(
        "Choose a stock",
        ranking["Ticker"].tolist()
    )

    stock = ranking[ranking["Ticker"] == selected_ticker].iloc[0]

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("AI Score", f"{stock['AI Score']} / 100")
    col2.metric("AI Opinion", stock["Simple View"])
    col3.metric("Risk Level", stock["Risk Level"])
    col4.metric("Business Category", stock["Sector"])

    st.markdown("### Why the AI may like this stock")

    reasons = []

    if stock["Return_252d"] > 0.20:
        reasons.append("The stock has had strong performance over the last year.")
    elif stock["Return_252d"] < -0.10:
        reasons.append("The stock has struggled over the last year, so the AI may be looking for a rebound.")

    if stock["Revenue_Growth"] > 0.05:
        reasons.append("The company appears to be growing sales.")
    else:
        reasons.append("Sales growth does not look especially strong.")

    if stock["Earnings_Growth"] > 0.05:
        reasons.append("Company earnings appear to be improving.")
    else:
        reasons.append("Earnings growth looks weak or unclear.")

    if stock["ROE"] > 0.15:
        reasons.append("The company appears efficient at turning investor money into profit.")

    if stock["Volatility_20d"] > 0.04:
        reasons.append("Price movement has been high, so this may be a riskier pick.")

    for reason in reasons:
        st.write(f"- {reason}")

    st.markdown("### Plain-English numbers")

    detail_data = {
        "What it means": [
            "AI confidence this stock could be a top performer",
            "Past 1-year stock performance",
            "Recent price movement risk",
            "Sales growth",
            "Earnings growth",
            "Profit efficiency"
        ],
        "Value": [
            f"{stock['AI Score']} / 100",
            f"{stock['Return_252d']:.2%}",
            f"{stock['Volatility_20d']:.2%}",
            f"{stock['Revenue_Growth']:.2%}",
            f"{stock['Earnings_Growth']:.2%}",
            f"{stock['ROE']:.2%}"
        ]
    }

    st.dataframe(
        pd.DataFrame(detail_data),
        use_container_width=True,
        hide_index=True
    )

with tab3:
    st.subheader("Suggested diversified portfolio")
    st.caption("This limits how many stocks come from the same business category.")

    st.dataframe(
        portfolio[show_cols],
        use_container_width=True,
        hide_index=True
    )

    st.bar_chart(
        portfolio.set_index("Ticker")["AI Score"]
    )

with tab4:
    st.subheader("What the AI is paying attention to")

    friendly_names = {
        "Earnings_Growth": "Earnings growth",
        "Revenue_Growth": "Sales growth",
        "Debt_To_Equity": "Debt level",
        "Forward_PE": "Expected price vs earnings",
        "Return_252d": "Past 1-year performance",
        "Free_Cashflow": "Cash left after expenses",
        "ROE": "Profit efficiency",
        "Market_Cap": "Company size",
        "Volatility_20d": "Recent price risk",
        "MA50_vs_MA200": "Trend strength"
    }

    importance["Plain English"] = importance["Feature"].replace(friendly_names)

    st.dataframe(
        importance[["Plain English", "Importance"]],
        use_container_width=True,
        hide_index=True
    )

    st.bar_chart(
        importance.set_index("Plain English")["Importance"]
    )

st.download_button(
    "Download full stock ranking",
    ranking.to_csv(index=False),
    "stock_ranking.csv",
    "text/csv"
)