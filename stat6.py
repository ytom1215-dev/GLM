import streamlit as st
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib  # ← これを追加（グラフの日本語文字化けを直す魔法のライブラリ）

st.set_page_config(page_title="第6章 GLMの拡張", layout="wide")

st.title("第6章 GLMの拡張")
st.subheader("〜 データの分布に合ったモデルを選ぶ 〜")
st.markdown("""
世の中のデータは、必ずしも美しい「正規分布（ベルカーブ）」をしているとは限りません。
カウントデータ（0以上の整数）や二値データ（0 or 1）に対して通常の「直線回帰」を当てはめると、おかしな予測（マイナスの数や100%超え）をしてしまいます。
このアプリで、**ポアソン分布**と**二項分布**を用いたGLMの威力を体感しましょう。
""")

# タブを作成
tab1, tab2, tab3 = st.tabs([
    "📈 1. ポアソン回帰 (カウントデータ)", 
    "📊 2. ロジスティック回帰 (割合・二値データ)", 
    "📚 まとめ (GLMの3要素)"
])

# ==========================================
# タブ1: ポアソン回帰
# ==========================================
with tab1:
    st.header("ポアソン回帰：カウントデータは「マイナス」にならない")
    st.markdown("対象データ：**害虫の数、病斑の数、Webサイトのクリック数**（0以上の整数）")
    
    # データ生成
    np.random.seed(123)
    temps = np.repeat(np.arange(15, 32.5, 2.5), 5)
    lambda_val = np.exp(0.15 * temps - 1.5)
    counts = np.random.poisson(lambda_val)
    df_pois = pd.DataFrame({'Temp': temps, 'Count': counts})

    col1, col2 = st.columns([2, 1])

    with col2:
        st.info("🎓 **研修用チェックポイント**\n\n"
                "**① 直線の限界:** グレーの点線（通常の直線回帰）を見てください。X（温度）が小さくなると、予測値が**マイナスに突入**してしまいます。\n\n"
                "**② ポアソン回帰の強み:** 赤い実線（ポアソン回帰）は、Yが「絶対に0以上になる」という性質を保ちながら、データにうまくフィット（指数関数的なカーブ）しています。\n\n"
                "**③ 回帰係数の意味:** 説明変数が1増えると、目的変数は $\\exp(\\beta_1)$ 倍になります。")

        # モデル構築
        model_ols = smf.ols('Count ~ Temp', data=df_pois).fit()
        model_poi = smf.glm('Count ~ Temp', data=df_pois, family=sm.families.Poisson()).fit()
        
        b0, b1 = model_poi.params
        pval = model_poi.pvalues['Temp']
        
        st.success(f"**ポアソン回帰の係数 (β1)**: {b1:.4f}\n\n"
                   f"**P値**: {pval:.5e}\n\n"
                   f"**解釈**: 温度が1℃上がると、カウント数は約 **{np.exp(b1):.2f}倍** になる")

        st.markdown("**Pythonコード (Statsmodels)**")
        st.code("smf.glm('Y ~ X', data, family=sm.families.Poisson()).fit()", language="python")

    with col1:
        # プロット
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.stripplot(x='Temp', y='Count', data=df_pois, color="#27ae60", alpha=0.6, jitter=0.2, ax=ax)
        
        x_pred = np.linspace(10, 35, 100)
        y_ols = model_ols.predict(pd.DataFrame({'Temp': x_pred}))
        y_poi = model_poi.predict(pd.DataFrame({'Temp': x_pred}))
        
        # x_pred を実際の軸のスケールに合わせるための調整
        x_ticks = np.arange(15, 32.5, 2.5)
        ax.set_xticks(range(len(x_ticks)))
        ax.set_xticklabels(x_ticks)
        
        # 描画用のX座標変換
        x_plot = (x_pred - 15) / 2.5
        ax.plot(x_plot, y_ols, color="#7f8c8d", linestyle="--", label="通常の直線回帰 (OLS)")
        ax.plot(x_plot, y_poi, color="#e74c3c", linewidth=2, label="ポアソン回帰 (GLM)")
        ax.axhline(0, color='black', linewidth=0.8)
        
        ax.set_title("カウントデータのモデリング")
        ax.set_xlabel("温度 (Temp)")
        ax.set_ylabel("発生数 (Count)")
        ax.legend()
        st.pyplot(fig)


# ==========================================
# タブ2: ロジスティック回帰
# ==========================================
with tab2:
    st.header("ロジスティック回帰：割合や生データに「S字カーブ」を当てる")
    st.markdown("対象データ：**発芽率、コンバージョン率、購入の有無**（0〜1の割合、または 0 or 1 の生データ）")
    
    data_type = st.radio("データの見せ方を選択してください", 
                         ("0/1 生データ (各個体の結果)", "割合データ (グループごとの集計結果)"), 
                         horizontal=True)

    # データ生成 (25℃で 50/100 になる設定)
    temps_logi = [21, 23, 25, 27, 29]
    successes = [5, 15, 50, 85, 95]
    totals = [100, 100, 100, 100, 100]

    # 生データ展開
    temp_raw = []
    flag_raw = []
    for t, s, tot in zip(temps_logi, successes, totals):
        temp_raw.extend([t]*tot)
        flag_raw.extend([1]*s + [0]*(tot-s))
    df_raw = pd.DataFrame({'Temp': temp_raw, 'Flag': flag_raw})

    # 割合データ
    df_rate = pd.DataFrame({'Temp': temps_logi, 'Success': successes, 'Total': totals})
    df_rate['Rate'] = df_rate['Success'] / df_rate['Total']

    col1, col2 = st.columns([2, 1])

    with col2:
        if "生データ" in data_type:
            st.info("💡 **プロットの特徴 (生データ)**\n\n"
                    "点は必ず「Y=0（未発芽）」か「Y=1（発芽）」の上下両極端に分かれて打たれます。（重なりを防ぐため少し散らしています）\n\n"
                    "中央の【25℃】を見ると、発芽した50個が上、未発芽の50個が下に真っ二つに分かれています。")
            model_logi = smf.logit('Flag ~ Temp', data=df_raw).fit(disp=0)
            b0, b1 = model_logi.params
            pval = model_logi.pvalues['Temp']
            st.code("smf.logit('Y ~ X', data).fit()", language="python")
        else:
            st.info("💡 **プロットの特徴 (割合データ)**\n\n"
                    "点は計算された割合（0〜1の中間）に直接打たれます。\n\n"
                    "中央の【25℃】を見ると、100個中50個発芽なので、「Y=0.5」の高さに点が1つだけプロットされています。")
            model_logi = smf.glm('Rate ~ Temp', data=df_rate, family=sm.families.Binomial(), var_weights=df_rate['Total']).fit()
            b0, b1 = model_logi.params
            pval = model_logi.pvalues['Temp']
            st.code("smf.glm('Rate ~ X', var_weights=Total,\n  family=sm.families.Binomial()).fit()", language="python")
        
        st.success(f"**回帰係数 (β1)**: {b1:.4f}\n\n**P値**: {pval:.5e}")
        st.warning("👉 **結論**: データの持ち方（見た目）が違っても、裏で計算される係数やP値、そして描かれるS字カーブは**完全に一致**します！")

    with col1:
        fig, ax = plt.subplots(figsize=(8, 5))
        
        if "生データ" in data_type:
            sns.stripplot(x='Temp', y='Flag', data=df_raw, color="#2980b9", alpha=0.1, jitter=0.2, ax=ax)
            ax.set_ylabel("発芽有無 (1=発芽, 0=未発芽)")
            ax.set_title("生データのロジスティック回帰")
        else:
            sns.scatterplot(x='Temp', y='Rate', data=df_rate, color="#2980b9", s=100, ax=ax)
            ax.set_ylabel("発芽割合 (Success / Total)")
            ax.set_title("割合データのロジスティック回帰")

        x_pred = np.linspace(20, 30, 100)
        y_pred = 1 / (1 + np.exp(-(b0 + b1 * x_pred)))
        
        x_ticks = temps_logi
        ax.set_xticks(range(len(x_ticks)))
        ax.set_xticklabels(x_ticks)
        
        x_plot = (x_pred - 21) / 2
        ax.plot(x_plot, y_pred, color="#e74c3c", linewidth=2, label="ロジスティック回帰 (S字カーブ)")
        ax.set_xlabel("温度 (Temp)")
        ax.legend()
        st.pyplot(fig)

# ==========================================
# タブ3: まとめ (GLMの3要素)
# ==========================================
with tab3:
    st.header("📚 まとめ：GLM（一般化線形モデル）の仕組み")
    st.markdown("""
    これまでの回帰分析はすべて**「GLM（一般化線形モデル）」**という大きな枠組みのバリエーションに過ぎません。
    データの性質に合わせて、**「①誤差の分布」**と**「②リンク関数」**のパーツを付け替えるだけで、様々なデータに対応できます。
    """)

    summary_df = pd.DataFrame({
        "モデル名": ["通常の直線回帰 (OLS)", "ポアソン回帰", "ロジスティック回帰"],
        "データの性質": ["連続値 (マイナスも可)", "カウントデータ (0以上の整数)", "二値・割合 (0〜1の範囲)"],
        "① 確率分布": ["正規分布", "ポアソン分布", "二項分布"],
        "② リンク関数": ["恒等リンク（そのまま）", "対数リンク（log）", "ロジットリンク（logit）"],
        "描かれる線": ["まっすぐな直線", "指数関数的なカーブ", "S字カーブ"]
    })
    
    st.table(summary_df.set_index("モデル名"))

    st.markdown("""
    ### 🔑 リンク関数とは？
    数式（モデル）が出力する値を、現実のデータ範囲に「変換（リンク）」する役割を持ちます。
    * **対数(log)リンク**: 予測値がマイナスにならないよう、右辺を $\\exp$ で持ち上げます。
    * **ロジット(logit)リンク**: 予測値が必ず 0〜1 の間に収まるよう、オッズの対数をとって変換します。
    
    Pythonの `statsmodels` では、`family=sm.families.Poisson()` のように「分布」を指定すると、自動的に適切なリンク関数がセットされる仕組みになっています。
    """)