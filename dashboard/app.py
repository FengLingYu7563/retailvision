"""RetailVision 儀表板（Streamlit，驗收 3.5）。

只透過 HTTP 呼叫 API（不直接連資料庫）：權限檢查都在 API 端，一般使用者只看得到自己的資料。
token 只存在這個瀏覽器分頁的 session_state，不寫進檔案或網址。

  API_BASE=http://127.0.0.1:8000 streamlit run dashboard/app.py
"""

from __future__ import annotations

import os
from datetime import date, datetime, timedelta
from typing import Any

import altair as alt
import httpx
import pandas as pd
import streamlit as st

API_BASE = os.environ.get("API_BASE", "http://127.0.0.1:8000").rstrip("/")
API = f"{API_BASE}/api/v1"
TIMEOUT = 60

st.set_page_config(page_title="RetailVision", layout="wide")


# ---------- API ----------


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


def call(method: str, path: str, **kwargs: Any) -> httpx.Response:
    headers = kwargs.pop("headers", {})
    if token := st.session_state.get("token"):
        headers["Authorization"] = f"Bearer {token}"
    r = httpx.request(method, f"{API}{path}", headers=headers, timeout=TIMEOUT, **kwargs)
    if r.status_code >= 400:
        try:
            err = r.json()["error"]
            code, message = err["code"], err["message"]
        except Exception:
            code, message = "HTTP_ERROR", r.text[:200]
        if r.status_code == 401 and path != "/auth/token":
            st.session_state.pop("token", None)  # token 過期：回到登入畫面
        raise ApiError(r.status_code, code, message)
    return r


def show_error(e: ApiError) -> None:
    st.error(f"{e.code}：{e.message}")


# ---------- 登入 ----------


def login_form() -> None:
    st.title("RetailVision")
    st.caption("零售商品影像辨識：單品辨識、貨架分析、以圖搜圖")
    with st.form("login"):
        username = st.text_input("帳號")
        password = st.text_input("密碼", type="password")
        if st.form_submit_button("登入"):
            try:
                r = call("POST", "/auth/token", data={"username": username, "password": password})
                st.session_state["token"] = r.json()["access_token"]
                st.session_state["username"] = username
                st.rerun()
            except ApiError as e:
                show_error(e)
            except httpx.HTTPError:
                st.error(f"連不到 API（{API_BASE}）")


# ---------- 頁面 ----------


def page_recognize() -> None:
    st.header("上傳辨識")
    kind = st.radio("類型", ["單品辨識", "貨架分析"], horizontal=True)
    strict = st.checkbox("品質不佳時拒絕（strict）")
    file = st.file_uploader(
        "圖片（jpg / png / webp，10 MB 以內）", type=["jpg", "jpeg", "png", "webp"]
    )
    if file is None or not st.button("送出辨識", type="primary"):
        return
    path = "/recognitions/product" if kind == "單品辨識" else "/recognitions/shelf"
    try:
        r = call(
            "POST",
            path,
            params={"strict": str(strict).lower()},
            files={"file": (file.name, file.getvalue(), file.type or "image/jpeg")},
        )
    except ApiError as e:
        show_error(e)
        return
    res = r.json()
    left, right = st.columns([3, 2])
    with left:
        img = call("GET", f"/recognitions/{res['id']}/annotated").content
        st.image(img, caption="標註圖", use_container_width=True)
    with right:
        q = res["quality"]
        if q["warnings"]:
            st.warning("品質警告：" + "、".join(q["warnings"]))
        st.metric("延遲", f"{res['latency_ms']:.0f} ms")
        st.caption(f"模型：{res['model_version']}")
        if res["type"] == "product":
            p = res["prediction"]
            st.metric("細類", p["fine"]["label"], f"{p['fine']['confidence']:.1%}")
            st.metric("粗類", p["coarse"]["label"], f"{p['coarse']['confidence']:.1%}")
            st.dataframe(pd.DataFrame(p["top_k"]), hide_index=True)
            m = res["catalog_match"]
            st.write(
                f"Catalog 配對：{m['name']}（相似度 {m['similarity']:.2f}）"
                if m
                else "Catalog 配對：無"
            )
        else:
            st.metric("商品數量", res["item_count"])
            st.metric("缺貨空位", len(res["gaps"]))
            st.metric("缺貨比例", f"{res['gap_ratio']:.1%}")
            st.metric("陳列密度", f"{res['density']:.1%}")


def page_history() -> None:
    st.header("歷史紀錄")
    c1, c2, c3, c4 = st.columns(4)
    kind = c1.selectbox("類型", ["全部", "product", "shelf"])
    label = c2.text_input("標籤（細類）")
    start = c3.date_input("起日", value=date.today() - timedelta(days=30))
    end = c4.date_input("迄日", value=date.today())
    size = 20
    page = st.number_input("頁數", min_value=1, value=1, step=1)
    params: dict[str, Any] = {
        "page": page,
        "size": size,
        "from": datetime.combine(start, datetime.min.time()).isoformat() + "Z",
        "to": datetime.combine(end, datetime.max.time()).isoformat() + "Z",
    }
    if kind != "全部":
        params["type"] = kind
    if label:
        params["label"] = label
    try:
        data = call("GET", "/recognitions", params=params).json()
    except ApiError as e:
        show_error(e)
        return
    st.caption(f"共 {data['total']} 筆")
    if not data["items"]:
        return
    df = pd.DataFrame(data["items"])
    df["created_at"] = pd.to_datetime(df["created_at"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    st.dataframe(
        df[
            [
                "created_at",
                "type",
                "top_label",
                "confidence",
                "model_version",
                "latency_ms",
                "quality_passed",
                "id",
            ]
        ],
        hide_index=True,
        use_container_width=True,
    )
    rid = st.selectbox("查看標註圖", ["（選擇一筆）"] + list(df["id"]))
    if rid != "（選擇一筆）":
        try:
            st.image(call("GET", f"/recognitions/{rid}/annotated").content, width=600)
        except ApiError as e:
            show_error(e)


def page_stats() -> None:
    st.header("統計")
    days = st.slider("統計天數", 1, 90, 30)
    try:
        s = call("GET", "/stats/summary", params={"days": days}).json()
    except ApiError as e:
        show_error(e)
        return
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("單品辨識次數", s["product"]["count"])
    avg = s["product"]["avg_confidence"]
    c2.metric("平均信心度", f"{avg:.1%}" if avg is not None else "無")
    c3.metric("貨架分析次數", s["shelf"]["count"])
    rate = s["shelf"]["stockout_rate"]
    c4.metric("缺貨率（有空位的貨架比例）", f"{rate:.1%}" if rate is not None else "無")

    left, right = st.columns(2)
    with left:
        st.subheader("類別分布（單品辨識）")
        if s["product"]["by_label"]:
            df = pd.DataFrame(s["product"]["by_label"])
            chart = (
                alt.Chart(df)
                .mark_bar()
                .encode(
                    x=alt.X("count:Q", title="次數"),
                    y=alt.Y("label:N", sort="-x", title="細類"),
                    tooltip=["label", "count", alt.Tooltip("avg_confidence:Q", format=".1%")],
                )
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("這段期間沒有單品辨識紀錄")
    with right:
        st.subheader("每日趨勢")
        if s["daily"]:
            df = pd.DataFrame(s["daily"])
            chart = (
                alt.Chart(df)
                .mark_line(point=True)
                .encode(
                    x=alt.X("date:T", title="日期"),
                    y=alt.Y("count:Q", title="次數"),
                    color=alt.Color("type:N", title="類型"),
                    tooltip=["date", "type", "count"],
                )
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("這段期間沒有紀錄")


def page_models() -> None:
    st.header("模型版本")
    try:
        models = call("GET", "/models").json()
    except ApiError as e:
        show_error(e)
        return
    rows = []
    for m in models:
        metrics = m["metrics"] or {}
        rows.append(
            {
                "名稱": m["name"],
                "任務": m["task"],
                "使用中": "是" if m["is_active"] else "",
                "細類 top-1": metrics.get("fine_top1"),
                "mAP50": metrics.get("onnx_mAP50") or metrics.get("mAP50"),
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.caption("切換使用中的分類模型需要 admin 權限，不需要重啟服務。")
    choices = [m["name"] for m in models if m["task"] == "classification"]
    target = st.selectbox("切換分類模型", choices)
    if st.button("切換"):
        try:
            r = call("PUT", "/models/active", json={"name": target})
            st.success(
                f"已切換為 {r.json()['active']}（重新計算 {r.json()['reembedded']} 個參考圖向量）"
            )
        except ApiError as e:
            show_error(e)


# ---------- 主程式 ----------


def main() -> None:
    if "token" not in st.session_state:
        login_form()
        return
    with st.sidebar:
        st.write(f"登入身分：{st.session_state.get('username', '')}")
        page = st.radio("頁面", ["上傳辨識", "歷史紀錄", "統計", "模型版本"])
        if st.button("登出"):
            st.session_state.clear()
            st.rerun()
    {
        "上傳辨識": page_recognize,
        "歷史紀錄": page_history,
        "統計": page_stats,
        "模型版本": page_models,
    }[page]()


main()
