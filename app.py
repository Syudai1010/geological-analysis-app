import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.cluster import KMeans
import streamlit as st
from io import BytesIO

plt.rcParams["font.family"] = "Meiryo"

# 土質区分小分類から土質区分大分類を判断（末尾の単語をチェック）
def classify_soil(soil_subtype):
    if not isinstance(soil_subtype, str):
        return '不明'
    last_word = soil_subtype.split()[-1]
    if last_word.endswith('礫'):
        return '礫質土'
    elif last_word.endswith('礫質土'):
        return '礫質土'
    elif last_word.endswith('砂'):
        return '砂質土'
    elif last_word.endswith('シルト') or last_word.endswith('粘土'):
        return '粘性土'
    elif last_word.endswith('火山灰'):
        return '火山灰'
    elif last_word.endswith('岩'):
        return '岩'
    return '不明'

# 大分類を簡略化（g, s, c）にする関数
def simplify_classification(soil_class):
    mapping = {
        '礫質土': 'g',  # g for gravel
        '砂質土': 's',  # s for sand
        '粘性土': 'c',  # c for cohesive soil (clay/silt)
        '火山灰': 'v',  # v for volcanic ash
        '岩': 'r',      # r for rock
        '不明': 'u'     # u for unknown
    }
    return mapping.get(soil_class, 'u')  # Default to 'u' if class is not found

# ニューラルネットの定義
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        if out.shape[1] > 1:
            out = self.softmax(out)
        return out

def main():
    st.title("地層分析ツール")

    mode = st.selectbox("モードを選択してください", ["学習モード", "予測モード", "地層図作成モード", "境界線モード"])
    n_value_mode = st.checkbox("N値モードを使用する")

    uploaded_file = st.file_uploader("ファイルを選択してください", type=['xlsx'])
    if uploaded_file is not None:
        xls = pd.ExcelFile(uploaded_file)
        sheet_names = xls.sheet_names
        sheet_name = st.selectbox("シートを選択してください", sheet_names)
        df = pd.read_excel(uploaded_file, sheet_name=sheet_name)

        # N値を50でクリップ
        if 'N値' in df.columns:
            df['N値'] = pd.to_numeric(df['N値'], errors='coerce').clip(upper=50)
            df['N値'] = df['N値'].fillna(0)  # 欠損値を0で埋める（必要に応じて変更）
        else:
            st.warning("'N値'列が存在しません。")

        # 説明変数と目的変数の選択
        columns = df.columns.tolist()
        explanation_var = st.selectbox("説明変数を選択してください（デフォルト：土質区分小分類）", columns, index=columns.index('土質区分小分類') if '土質区分小分類' in columns else 0)
        target_var = st.selectbox("目的変数を選択してください（デフォルト：地層区分_土質情報）", columns, index=columns.index('地層区分_土質情報') if '地層区分_土質情報' in columns else 0)

        if st.button("実行"):
            # 土質区分の大分類を生成
            df['土質区分大分類'] = df[explanation_var].apply(classify_soil)

            # 土質区分大分類の簡略化（g, s, cなどに変換）
            df['土質区分大分類簡略'] = df['土質区分大分類'].apply(simplify_classification)

            if n_value_mode and 'N値' not in df.columns:
                st.warning("'N値'列が存在しません。")
                return

            # 目的変数
            y = df[target_var].values
            target_le = LabelEncoder()
            y = target_le.fit_transform(y)

            # データフレームとその他のデータをセッション状態に保存
            st.session_state['df'] = df
            st.session_state['y'] = y
            st.session_state['n_value_mode'] = n_value_mode
            st.session_state['target_le'] = target_le

    # データフレームがセッションに保存されているか確認
    if 'df' in st.session_state:
        df = st.session_state['df']
        y = st.session_state['y']
        n_value_mode = st.session_state['n_value_mode']
        target_le = st.session_state['target_le']

        if mode == "学習モード":
            learning_mode(df, y, n_value_mode, target_le)
        elif mode == "予測モード":
            prediction_mode(df, y, n_value_mode)
            # 予測結果をセッションに保存
            st.session_state['df'] = df
        elif mode == "地層図作成モード":
            geological_map_mode(df)
        elif mode == "境界線モード":
            boundary_line_mode()
    else:
        st.warning("ファイルをアップロードし、実行ボタンを押してください。")

def learning_mode(df, y, n_value_mode, target_le):
    # One-hot encoding of '土質区分大分類簡略'
    one_hot_columns = pd.get_dummies(df['土質区分大分類簡略'], prefix='soil_type')
    # データ型をfloatに変換
    one_hot_columns = one_hot_columns.astype(float)

    if n_value_mode:
        X = pd.concat([one_hot_columns, df[['N値']]], axis=1)
    else:
        X = one_hot_columns

    # 特徴量のカラム名を保存
    feature_columns = X.columns.tolist()

    # 学習データとテストデータの分割
    X_train, X_test, y_train, y_test = train_test_split(X.values, y, test_size=0.2, random_state=42)
    input_size = X_train.shape[1]
    hidden_size = 100
    output_size = len(np.unique(y))
    model = NeuralNet(input_size, hidden_size, output_size)

    # 損失関数と最適化
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # 学習データをTensorに変換
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)

    # モデルの学習
    num_epochs = 1000
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

    # モデルとエンコーダーを保存
    joblib.dump(model, 'trained_model.pkl')
    joblib.dump(feature_columns, 'feature_columns.pkl')
    joblib.dump(target_le, 'target_label_encoder.pkl')
    st.success("モデルを保存しました！")

def prediction_mode(df, y_true, n_value_mode):
    # 予測モード
    try:
        model = joblib.load('trained_model.pkl')
        feature_columns = joblib.load('feature_columns.pkl')
        target_le = joblib.load('target_label_encoder.pkl')
    except FileNotFoundError:
        st.warning("モデルが見つかりません。まず学習モードでモデルを作成してください。")
        return

    # One-hot encoding of '土質区分大分類簡略'
    one_hot_columns = pd.get_dummies(df['土質区分大分類簡略'], prefix='soil_type')
    # データ型をfloatに変換
    one_hot_columns = one_hot_columns.astype(float)

    if n_value_mode:
        if 'N値' not in df.columns:
            st.warning("'N値'列が存在しません。")
            return
        X = pd.concat([one_hot_columns, df[['N値']]], axis=1)
    else:
        X = one_hot_columns

    # 学習時の特徴量と同じカラムを揃える
    for col in feature_columns:
        if col not in X.columns:
            X[col] = 0
    X = X[feature_columns]

    # データ型をfloatに変換
    X = X.astype(float)

    # 説明変数をTensorに変換して予測
    X_tensor = torch.FloatTensor(X.values)
    with torch.no_grad():
        model.eval()
        predictions = model(X_tensor).numpy()
        predicted_classes = np.argmax(predictions, axis=1)
        predicted_labels = target_le.inverse_transform(predicted_classes)

    # 予測結果をデータフレームに追加
    df['予測結果'] = predicted_labels
    df['予測結果_encoded'] = predicted_classes

    # コンター図の作成
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # クラスとカラーマップの設定
    classes = target_le.classes_
    n_classes = len(classes)
    cmap = plt.get_cmap('viridis', n_classes)
    norm = plt.Normalize(vmin=0, vmax=n_classes - 1)

    scatter_pred = ax[0].scatter(df['距離'], df['標高'], c=predicted_classes, cmap=cmap, norm=norm)
    ax[0].set_title('予測結果')

    for i, row in df.iterrows():
        ax[0].text(row['距離'], row['標高'], f"{predicted_labels[i]}", fontsize=9)

    y_true_encoded = y_true  # Already encoded
    scatter_true = ax[1].scatter(df['距離'], df['標高'], c=y_true_encoded, cmap=cmap, norm=norm)
    ax[1].set_title('教師データ')

    true_labels = target_le.inverse_transform(y_true_encoded)
    for i, row in df.iterrows():
        ax[1].text(row['距離'], row['標高'], f"{true_labels[i]}", fontsize=9)

    # カラーバーの設定
    cbar_pred = fig.colorbar(scatter_pred, ax=ax[0], ticks=np.arange(n_classes))
    cbar_pred.ax.set_yticklabels(classes)
    cbar_true = fig.colorbar(scatter_true, ax=ax[1], ticks=np.arange(n_classes))
    cbar_true.ax.set_yticklabels(classes)

    # 正解率の表示
    accuracy = np.mean(predicted_classes == y_true_encoded)
    st.info(f"正解率: {accuracy * 100:.2f}%")

    st.pyplot(fig)

    # 予測結果を保存
    df.to_csv('predicted_results.csv', index=False)
    st.success("予測結果を保存しました: predicted_results.csv")

def geological_map_mode(df):
    # 地層図作成モードの処理
    if df is not None and '予測結果' in df.columns:
        required_columns = ['土質区分小分類', '距離', '標高', '予測結果', 'N値']
        if all(col in df.columns for col in required_columns):
            # コンター図1（予測結果）
            fig, ax = plt.subplots(figsize=(12, 6))

            # ラベルエンコーダーとカラーマップの設定
            label_encoder = LabelEncoder()
            label_codes = label_encoder.fit_transform(df['予測結果'])
            classes = label_encoder.classes_
            n_classes = len(classes)
            cmap = plt.get_cmap('viridis', n_classes)
            norm = plt.Normalize(vmin=0, vmax=n_classes - 1)

            scatter = ax.scatter(df['距離'], df['標高'], c=label_codes, cmap=cmap, norm=norm)
            for i, row in df.iterrows():
                ax.text(row['距離'], row['標高'], f"{row['予測結果']}", fontsize=9)

            cbar = fig.colorbar(scatter, ax=ax, ticks=np.arange(n_classes))
            cbar.ax.set_yticklabels(classes)
            ax.set_xlabel('距離')
            ax.set_ylabel('標高')
            st.pyplot(fig)

            # 境界面の検出
            df['N値'] = df['N値'].clip(upper=50)
            boundary_x = []
            boundary_y = []
            grouped = df.groupby('距離')
            for name, group in grouped:
                mask = group['標高'] <= -10
                group = group[mask]
                if len(group) > 1:
                    n_diff = group['N値'].diff().fillna(0)
                    boundary_mask = n_diff >= 15
                    if boundary_mask.any():
                        first_boundary_idx = boundary_mask.idxmax()
                        boundary_x.append(group['距離'].loc[first_boundary_idx])
                        boundary_y.append(group['標高'].loc[first_boundary_idx])

            # 境界面を基にした層の分類
            def add_layer_prefix(row, boundary_x, boundary_y):
                if len(boundary_x) > 0:
                    boundary_level = np.interp(row['距離'], boundary_x, boundary_y)
                    if row['標高'] > boundary_level:
                        if row['予測結果'][0] in ['D', 'B']:
                            return 'A' + row['予測結果'][1:]
                        else:
                            return 'A' + row['予測結果']
                    else:
                        if row['予測結果'][0] in ['A', 'B']:
                            return 'D' + row['予測結果'][1:]
                        else:
                            return 'D' + row['予測結果']
                return row['予測結果']

            df['分類結果'] = df.apply(add_layer_prefix, axis=1, boundary_x=boundary_x, boundary_y=boundary_y)

            # コンター図2（分類結果）
            fig, ax = plt.subplots(figsize=(12, 6))

            # ラベルエンコーダーとカラーマップの設定
            label_encoder = LabelEncoder()
            label_codes = label_encoder.fit_transform(df['分類結果'])
            classes = label_encoder.classes_
            n_classes = len(classes)
            cmap = plt.get_cmap('coolwarm', n_classes)
            norm = plt.Normalize(vmin=0, vmax=n_classes - 1)

            scatter = ax.scatter(df['距離'], df['標高'], c=label_codes, cmap=cmap, norm=norm)
            for i, row in df.iterrows():
                ax.text(row['距離'], row['標高'], f"{row['分類結果']}", fontsize=9)

            cbar = fig.colorbar(scatter, ax=ax, ticks=np.arange(n_classes))
            cbar.ax.set_yticklabels(classes)
            ax.set_xlabel('距離')
            ax.set_ylabel('標高')

            if boundary_x and boundary_y:
                ax.plot(boundary_x, boundary_y, color='black', linewidth=2, linestyle='--', zorder=3)
            st.pyplot(fig)

            # 盛土をBに変更
            def replace_with_b(row):
                if isinstance(row['土質区分小分類'], str) and row['土質区分小分類'].startswith('盛土'):
                    return 'B' + row['分類結果'][1:]
                return row['分類結果']

            df['分類結果B'] = df.apply(replace_with_b, axis=1)

            # コンター図3（分類結果B）
            fig, ax = plt.subplots(figsize=(12, 6))

            label_encoder = LabelEncoder()
            label_codes = label_encoder.fit_transform(df['分類結果B'])
            classes = label_encoder.classes_
            n_classes = len(classes)
            cmap = plt.get_cmap('coolwarm', n_classes)
            norm = plt.Normalize(vmin=0, vmax=n_classes - 1)

            scatter = ax.scatter(df['距離'], df['標高'], c=label_codes, cmap=cmap, norm=norm)
            for i, row in df.iterrows():
                ax.text(row['距離'], row['標高'], f"{row['分類結果B']}", fontsize=9)

            cbar = fig.colorbar(scatter, ax=ax, ticks=np.arange(n_classes))
            cbar.ax.set_yticklabels(classes)
            ax.set_xlabel('距離')
            ax.set_ylabel('標高')
            st.pyplot(fig)

            # 火山灰や岩の処理
            def replace_with_v_or_alt(row):
                if isinstance(row['土質区分小分類'], str):
                    if row['土質区分小分類'].endswith('火山灰'):
                        if len(row['分類結果B']) > 1:
                            return row['分類結果B'][:-1] + 'v'
                        else:
                            return row['分類結果B']
                    elif row['土質区分小分類'].endswith('岩'):
                        return 'Alt'
                return row['分類結果B']

            df['分類結果vAlt'] = df.apply(replace_with_v_or_alt, axis=1)

            # コンター図4（分類結果vAlt）
            fig, ax = plt.subplots(figsize=(12, 6))

            label_encoder = LabelEncoder()
            label_codes = label_encoder.fit_transform(df['分類結果vAlt'])
            classes = label_encoder.classes_
            n_classes = len(classes)
            cmap = plt.get_cmap('coolwarm', n_classes)
            norm = plt.Normalize(vmin=0, vmax=n_classes - 1)

            scatter = ax.scatter(df['距離'], df['標高'], c=label_codes, cmap=cmap, norm=norm)
            for i, row in df.iterrows():
                ax.text(row['距離'], row['標高'], f"{row['分類結果vAlt']}", fontsize=9)

            cbar = fig.colorbar(scatter, ax=ax, ticks=np.arange(n_classes))
            cbar.ax.set_yticklabels(classes)
            ax.set_xlabel('距離')
            ax.set_ylabel('標高')
            st.pyplot(fig)

            # 最終データをベクトルデータとして保存 (x, y, z形式)
            vector_data = df[['距離', '標高', '分類結果vAlt']]
            vector_data.columns = ['x', 'y', 'z']  # Rename columns to x, y, z
            vector_data.to_csv('geological_vector_data.csv', index=False)
            st.success("ベクトルデータを保存しました: geological_vector_data.csv")

            # ダウンロードリンクの提供
            csv = vector_data.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="ベクトルデータをダウンロード",
                data=csv,
                file_name='geological_vector_data.csv',
                mime='text/csv',
            )
        else:
            st.warning("指定されたカラムがデータに存在しません。'土質区分小分類', '距離', '標高', '予測結果', 'N値'が必要です。")
    else:
        st.warning("予測結果がありません。まず予測モードで予測を実行してください。")


def boundary_line_mode():
    # 境界線モードの処理
    csv_file_path = 'geological_vector_data.csv'  # ベクトルデータのパス
    try:
        data = pd.read_csv(csv_file_path)

        # 列を指定
        x = data['x']
        y = data['y']
        labels = data['z']

        # ラベルをエンコード
        label_encoder = LabelEncoder()
        z_numeric = label_encoder.fit_transform(labels)
        unique_labels = label_encoder.classes_

        # カラーマップを作成
        n_classes = len(unique_labels)
        cmap = plt.get_cmap('tab10', n_classes)
        colors = cmap(range(n_classes))
        label_colors = dict(zip(unique_labels, colors))

        # SVMの分類器を作成
        clf = SVC(kernel='rbf', C=30, gamma=1.5)
        clf.fit(data[['x', 'y']], z_numeric)

        # メッシュを作成
        x_min, x_max = x.min() - 0.05, x.max() + 0.05
        y_min, y_max = y.min() - 1, y.max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

        # 境界を予測
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        # プロット
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.contour(xx, yy, Z, levels=np.arange(Z.max() + 1) - 0.5, colors='k', linewidths=0.5)  # 境界線を細く設定

        # 各ラベルに対してプロット
        for i, label in enumerate(unique_labels):
            indices = labels == label
            ax.scatter(x[indices], y[indices], color=label_colors[label], edgecolor='k', label=f'{label}', s=50)

        ax.set_xlim(x_min, x_max)
        ax.set_xlabel('距離')
        ax.set_ylabel('標高')
        ax.set_title('SVMによる境界線とデータポイント')
        ax.legend()

        st.pyplot(fig)

    except FileNotFoundError:
        st.warning("ベクトルデータファイルが見つかりません。まず地層図作成モードでベクトルデータを作成してください。")

if __name__ == '__main__':
    main()
