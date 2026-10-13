"""Static HTML Report Generator for Photo Advisor Offline Run.

Renders a self-contained static HTML page displaying:
- Fixed reference input image
- Side-by-side comparison cards for No-op, Rule v0, and Imported Candidate(s)
- Badges for status, export eligibility, requested vs effective recipes, delta clipping, and logs.
No external server required.
"""

import html
from typing import Any, Dict, List


def generate_static_html(
    manifest_data: Dict[str, Any],
    output_html_path: str,
) -> None:
    """Generate self-contained static HTML report."""
    meta = manifest_data.get("metadata", {})
    records = manifest_data.get("candidates", [])

    cards_html = []
    for r in records:
        cand_id = html.escape(str(r.get("candidate_id", "")))
        source_name = html.escape(str(r.get("source_name", "")))
        status = html.escape(str(r.get("status", "unknown")))
        preview_file = html.escape(str(r.get("preview_filename", "")))
        is_real = r.get("is_real_model", False)
        real_badge = (
            '<span class="badge real">Real Model</span>'
            if is_real
            else '<span class="badge fixture">Offline / Fixture</span>'
        )

        status_class = "status-success" if status == "success" else "status-fallback"
        status_badge = f'<span class="badge {status_class}">{status.upper()}</span>'

        # Export eligibility badge (B2: distinguish dimension <400px vs fallback/guardrail/rejection)
        is_export_eligible = r.get("export_eligible", True)
        if is_export_eligible:
            eligible_badge = '<span class="badge eligible">Export Eligible</span>'
        else:
            ineligible_reason = str(r.get("export_ineligible_reason") or "")
            input_eligible = meta.get("input_export_eligible", True)

            # Determine precise badge reason
            if not input_eligible or "dimension" in ineligible_reason or "<400" in ineligible_reason:
                badge_text = "Preview Only (&lt;400px)"
            elif status == "fallback_noop" or "guardrail" in ineligible_reason or "fallback" in ineligible_reason:
                badge_text = "Preview Only (Guardrail Fallback)"
            elif status.startswith("rejected") or "rejected" in ineligible_reason or "error" in ineligible_reason:
                badge_text = "Preview Only (Rejected)"
            else:
                badge_text = "Preview Only"
            eligible_badge = f'<span class="badge ineligible">{badge_text}</span>'

        req_recipe = r.get("requested_recipe", {})
        eff_recipe = r.get("effective_recipe", {})
        notes = html.escape(str(r.get("notes", "")))
        out_sha = html.escape(str(r.get("output_sha256", ""))[:16]) + "..." if r.get("output_sha256") else "N/A"

        delta_clip = r.get("delta_clipping", {})
        du = delta_clip.get("delta_under", 0.0)
        do = delta_clip.get("delta_over", 0.0)
        clip_str = f"Under: {du:+.4f} | Over: {do:+.4f}"

        attempted_clip = r.get("attempted_clipping")
        attempted_str = ""
        if attempted_clip and (status == "fallback_noop" or attempted_clip != delta_clip):
            au = attempted_clip.get("delta_under", 0.0)
            ao = attempted_clip.get("delta_over", 0.0)
            attempted_str = f'<div><strong>Attempted Delta:</strong> <code>Under: {au:+.4f} | Over: {ao:+.4f}</code></div>'

        fallback_reason = r.get("fallback_reason")
        fallback_html = (
            f'<div class="alert-box warning"><strong>Fallback Reason:</strong> {html.escape(str(fallback_reason))}</div>'
            if fallback_reason
            else ""
        )

        clamp_warnings = r.get("clamp_warnings", [])
        warnings_html = ""
        if clamp_warnings:
            items = "".join(f"<li>{html.escape(w)}</li>" for w in clamp_warnings)
            warnings_html = f'<div class="alert-box info"><strong>Clamp Warnings:</strong><ul>{items}</ul></div>'

        card = f"""
        <div class="card">
            <div class="card-header">
                <h3>{source_name} <small>({cand_id})</small></h3>
                <div class="badges">
                    {status_badge}
                    {real_badge}
                    {eligible_badge}
                </div>
            </div>
            <div class="card-body">
                <div class="img-container">
                    <img src="{preview_file}" alt="Preview for {cand_id}" loading="lazy" />
                </div>
                <div class="meta-section">
                    <h4>Effective Recipe</h4>
                    <pre>{html.escape(str(eff_recipe))}</pre>
                    <h4>Requested Recipe</h4>
                    <pre class="dim">{html.escape(str(req_recipe))}</pre>
                    <div class="stats-grid">
                        <div><strong>Effective Delta:</strong> <code>{clip_str}</code></div>
                        {attempted_str}
                        <div><strong>Output SHA256:</strong> <code>{out_sha}</code></div>
                    </div>
                    {fallback_html}
                    {warnings_html}
                    {f'<div class="notes"><strong>Ghi chú:</strong> {notes}</div>' if notes else ''}
                </div>
            </div>
        </div>
        """
        cards_html.append(card)

    vlm_summary = meta.get("vlm_summary")
    vlm_section_html = ""
    if vlm_summary:
        m_id = html.escape(str(vlm_summary.get("model_id", "VLM")))
        v_status = html.escape(str(vlm_summary.get("status", "unknown")))
        analysis = vlm_summary.get("analysis") or {}
        framing = html.escape(str(analysis.get("framing", "N/A")))
        subject = html.escape(str(analysis.get("subject_description", "N/A")))
        background = html.escape(str(analysis.get("background_description", "N/A")))
        lighting = html.escape(str(analysis.get("lighting_condition", "N/A")))
        uncertainty = html.escape(str(analysis.get("uncertainty") or "Không có (rõ ràng)"))

        keep_orig = vlm_summary.get("keep_original") or {}
        is_keep = keep_orig.get("recommended", False)
        keep_rec = "NÊN GIỮ GỐC" if is_keep else "NÊN CHỈNH SỬA"
        keep_reason = html.escape(str(keep_orig.get("reason", "N/A")))

        retake = vlm_summary.get("retake_advice")
        retake_text = html.escape(str(retake)) if retake else "Không có lời khuyên chụp lại."

        vlm_section_html = f"""
        <section class="vlm-assessment-section" style="background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); padding: 20px; margin-bottom: 24px;">
            <h2 style="font-size: 1.3rem; margin-bottom: 12px; color: var(--primary);">Đánh giá của Trợ lý VLM ({m_id} &bull; {v_status.upper()})</h2>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; font-size: 0.9rem;">
                <div style="background: rgba(15, 23, 42, 0.6); padding: 12px; border-radius: 6px; border: 1px solid var(--border);">
                    <h4 style="color: var(--primary); margin-bottom: 6px; text-transform: uppercase; font-size: 0.8rem;">Phân tích bối cảnh & ánh sáng</h4>
                    <p><strong>Cỡ cảnh:</strong> {framing}</p>
                    <p><strong>Chủ thể:</strong> {subject}</p>
                    <p><strong>Hậu cảnh:</strong> {background}</p>
                    <p><strong>Ánh sáng:</strong> {lighting}</p>
                    <p><strong>Độ không chắc chắn:</strong> {uncertainty}</p>
                </div>
                <div style="background: rgba(15, 23, 42, 0.6); padding: 12px; border-radius: 6px; border: 1px solid var(--border);">
                    <h4 style="color: var(--primary); margin-bottom: 6px; text-transform: uppercase; font-size: 0.8rem;">Đánh giá giữ gốc</h4>
                    <p><strong>Khuyến nghị:</strong> <span class="badge {'status-success' if is_keep else 'fixture'}">{keep_rec}</span></p>
                    <p style="margin-top: 6px; color: var(--text-muted);">{keep_reason}</p>
                </div>
                <div style="background: rgba(15, 23, 42, 0.6); padding: 12px; border-radius: 6px; border: 1px solid var(--border);">
                    <h4 style="color: #fbbf24; margin-bottom: 6px; text-transform: uppercase; font-size: 0.8rem;">Tư vấn chụp lại cho lần sau</h4>
                    <p style="color: var(--text-muted);">{retake_text}</p>
                    <small style="color: #94a3b8; font-size: 0.75rem; display: block; margin-top: 6px;">&bull; Tư vấn thuần văn bản; không dựng preview giả lập cho thao tác chụp lại.</small>
                </div>
            </div>
        </section>
        """

    html_content = f"""<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Photo Advisor Phase 02B - VLM Integration Report</title>
    <style>
        :root {{
            --bg: #0f172a;
            --surface: #1e293b;
            --surface-hover: #334155;
            --text: #f8fafc;
            --text-muted: #94a3b8;
            --primary: #38bdf8;
            --success: #22c55e;
            --warning: #f59e0b;
            --danger: #ef4444;
            --border: #334155;
            --radius: 12px;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            background: var(--bg);
            color: var(--text);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            line-height: 1.5;
            padding: 24px;
        }}
        .container {{ max-width: 1300px; margin: 0 auto; }}
        header {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            padding: 24px;
            margin-bottom: 24px;
        }}
        h1 {{ font-size: 1.8rem; margin-bottom: 8px; color: var(--primary); }}
        .header-meta {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 12px;
            font-size: 0.9rem;
            color: var(--text-muted);
            margin-top: 12px;
        }}
        .header-meta div strong {{ color: var(--text); }}
        .reference-section {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            padding: 20px;
            margin-bottom: 24px;
            text-align: center;
        }}
        .reference-section h2 {{ font-size: 1.3rem; margin-bottom: 12px; color: var(--primary); }}
        .reference-img-container img {{
            max-width: 100%;
            max-height: 400px;
            border-radius: 8px;
            border: 1px solid var(--border);
            object-fit: contain;
        }}
        .comparison-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
            gap: 20px;
        }}
        .card {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            overflow: hidden;
            display: flex;
            flex-direction: column;
        }}
        .card-header {{
            padding: 16px;
            border-bottom: 1px solid var(--border);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .card-header h3 {{ font-size: 1.1rem; }}
        .card-header small {{ color: var(--text-muted); font-size: 0.8rem; }}
        .card-body {{ padding: 16px; display: flex; flex-direction: column; gap: 12px; flex: 1; }}
        .img-container {{
            background: #020617;
            border-radius: 8px;
            text-align: center;
            padding: 8px;
            min-height: 260px;
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        .img-container img {{
            max-width: 100%;
            max-height: 280px;
            border-radius: 6px;
            object-fit: contain;
        }}
        .meta-section {{ font-size: 0.85rem; }}
        .meta-section h4 {{ font-size: 0.85rem; text-transform: uppercase; color: var(--primary); margin: 8px 0 4px; }}
        pre {{
            background: #090d16;
            padding: 8px;
            border-radius: 6px;
            font-size: 0.75rem;
            overflow-x: auto;
            border: 1px solid #1e293b;
        }}
        pre.dim {{ opacity: 0.75; }}
        .stats-grid {{ margin-top: 8px; font-size: 0.8rem; display: flex; flex-direction: column; gap: 4px; }}
        .notes {{ margin-top: 8px; padding: 8px; background: rgba(56, 189, 248, 0.08); border-radius: 6px; font-size: 0.8rem; }}
        .alert-box {{ margin-top: 8px; padding: 8px 12px; border-radius: 6px; font-size: 0.8rem; }}
        .alert-box.warning {{ background: rgba(245, 158, 11, 0.15); border: 1px solid var(--warning); color: #fbbf24; }}
        .alert-box.info {{ background: rgba(56, 189, 248, 0.15); border: 1px solid var(--primary); color: #7dd3fc; }}
        .alert-box ul {{ padding-left: 16px; margin-top: 4px; }}
        .badges {{ display: flex; flex-wrap: wrap; gap: 6px; }}
        .badge {{
            padding: 3px 8px;
            border-radius: 999px;
            font-size: 0.7rem;
            font-weight: 600;
            text-transform: uppercase;
        }}
        .badge.status-success {{ background: rgba(34, 197, 94, 0.2); color: var(--success); }}
        .badge.status-fallback {{ background: rgba(245, 158, 11, 0.2); color: var(--warning); }}
        .badge.real {{ background: rgba(56, 189, 248, 0.2); color: var(--primary); }}
        .badge.fixture {{ background: rgba(148, 163, 184, 0.2); color: var(--text-muted); }}
        .badge.eligible {{ background: rgba(34, 197, 94, 0.15); color: #4ade80; border: 1px solid rgba(34, 197, 94, 0.4); }}
        .badge.ineligible {{ background: rgba(239, 68, 68, 0.15); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.4); }}
        footer {{
            margin-top: 32px;
            text-align: center;
            font-size: 0.8rem;
            color: var(--text-muted);
            border-top: 1px solid var(--border);
            padding-top: 16px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Photo Advisor Phase 02A - Offline Pipeline Verification</h1>
            <p>Trang minh chứng so sánh trực quan độc lập (No-op vs Rule v0 vs Fixture). Không cần web server.</p>
            <div class="header-meta">
                <div><strong>Run ID:</strong> {html.escape(str(meta.get("run_id", "")))}</div>
                <div><strong>Nguồn ảnh:</strong> {html.escape(str(meta.get("input_source", "synthetic")))}</div>
                <div><strong>Mục tiêu (Goal):</strong> {html.escape(str(meta.get("goal", "")))}</div>
                <div><strong>Runtime Env:</strong> {html.escape(str(meta.get("environment", "")))}</div>
                <div><strong>Thư viện thực tế:</strong> {html.escape(str(meta.get("dependencies", "")))}</div>
                <div><strong>Input SHA256:</strong> <code>{html.escape(str(meta.get("input_sha256", ""))[:16])}...</code></div>
                <div><strong>Thời gian chạy:</strong> {html.escape(str(meta.get("timestamp", "")))}</div>
            </div>
        </header>

        <section class="reference-section">
            <h2>Ảnh gốc tham chiếu chuẩn hóa (Fixed Reference)</h2>
            <div class="reference-img-container">
                <img src="reference_input.png" alt="Reference Input Image" />
            </div>
            <p style="margin-top: 8px; font-size: 0.85rem; color: var(--text-muted);">
                Đã chuẩn hóa EXIF orientation upright và chuyển đổi ICC sang sRGB chuẩn 8-bit.
            </p>
        </section>

        {vlm_section_html}

        <section class="comparison-grid">
            {''.join(cards_html)}
        </section>

        <footer>
            Đồ án Trợ lý Cải thiện Ảnh &bull; Vòng 02B VLM Integration &bull; Antigravity Pair-Programming
        </footer>
    </div>
</body>
</html>
"""
    with open(output_html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
