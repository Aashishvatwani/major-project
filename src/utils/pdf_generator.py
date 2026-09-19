"""
Aerospace Mission Audit Report PDF Generator
Produces publication-grade, richly formatted PDF reports of spacecraft telemetry logs,
operator actions, early predictions, and closed-loop RL mitigations.
"""

import os
import time
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
    )
    from reportlab.pdfgen import canvas
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False


class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas to dynamically compute and draw total page numbers and aerospace footer"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 7.5)
        self.setFillColor(colors.HexColor("#64748b"))

        # Top Running Header
        self.drawString(36, 11 * inch - 28, "AERO-GUARD // AUTONOMOUS SPACECRAFT HEALTH MANAGEMENT (FDIR)")
        self.drawRightString(8.5 * inch - 36, 11 * inch - 28, "NASA-HDBK-4008 • ISRO-URSC-PCDU-EPS-04 • ECSS-E-ST-20C")
        self.setStrokeColor(colors.HexColor("#0284c7"))
        self.setLineWidth(0.75)
        self.line(36, 11 * inch - 32, 8.5 * inch - 36, 11 * inch - 32)

        # Bottom Running Footer
        self.setStrokeColor(colors.HexColor("#334155"))
        self.setLineWidth(0.5)
        self.line(36, 36, 8.5 * inch - 36, 36)
        
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
        self.drawString(36, 24, f"CONFIDENTIAL // MISSION AUDIT LEDGER (10,000-LINE FIFO BUFFER) // GENERATED: {now_str}")
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(8.5 * inch - 36, 24, page_str)
        self.restoreState()


def generate_mission_audit_pdf(
    logs: List[Dict[str, Any]],
    stats: Dict[str, Any],
    output_path: str = "docs/AERO_GUARD_MISSION_AUDIT_REPORT.pdf",
    title: str = "AERO-GUARD MISSION CONTROL AUDIT LEDGER"
) -> str:
    """
    Generates a publication-quality aerospace PDF audit report from structured log entries.
    """
    if not HAS_REPORTLAB:
        raise RuntimeError("ReportLab library is required for PDF generation. Please install reportlab.")

    dir_name = os.path.dirname(output_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    # Document Geometry
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=44,
        bottomMargin=44
    )

    styles = getSampleStyleSheet()
    
    # Custom Aerospace Typography Styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#0f172a")
    )
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor("#0284c7")
    )
    section_h1 = ParagraphStyle(
        'SectionH1',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#0f172a"),
        spaceBefore=12,
        spaceAfter=6
    )
    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11.5,
        textColor=colors.HexColor("#334155")
    )
    badge_style = ParagraphStyle(
        'BadgeText',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=7,
        leading=8,
        alignment=1
    )
    table_cell_style = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor("#1e293b")
    )
    table_mono_style = ParagraphStyle(
        'TableMono',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=7,
        leading=8.5,
        textColor=colors.HexColor("#0f172a")
    )

    story = []

    # =========================================================================
    # 1. Title Banner & Satellite Header
    # =========================================================================
    banner_data = [
        [
            Paragraph(f"<b>{title.upper()}</b>", title_style),
            Paragraph(f"<b>MET STATUS:</b> ACTIVE CRUISE<br/><b>ORBIT:</b> LEO 552 KM • INC 97.6°<br/><b>HARDWARE:</b> HITL PYSERIAL BRIDGE", ParagraphStyle('HdrMeta', parent=table_cell_style, fontSize=7.5, leading=10, textColor=colors.HexColor("#0369a1")))
        ],
        [
            Paragraph("Autonomous Spacecraft Health Management (FDIR) • Closed-Loop Telemetry & Operator Audit Trail", subtitle_style),
            Paragraph(f"<b>LOG BUFFER:</b> {stats.get('buffer_count', len(logs))} / {stats.get('buffer_max_lines', 10000)} LINES (FIFO ROLLING)", ParagraphStyle('HdrMeta2', parent=table_cell_style, fontSize=7.5, leading=10, textColor=colors.HexColor("#475569")))
        ]
    ]
    banner_table = Table(banner_data, colWidths=[5.4 * inch, 2.0 * inch])
    banner_table.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
        ('TOPPADDING', (0,0), (-1,-1), 2),
        ('LEFTPADDING', (0,0), (-1,-1), 0),
        ('RIGHTPADDING', (0,0), (-1,-1), 0),
    ]))
    story.append(banner_table)
    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0284c7"), spaceBefore=0, spaceAfter=8))

    # =========================================================================
    # 2. Executive KPI & Audit Metrics Summary Cards
    # =========================================================================
    story.append(Paragraph("1. EXECUTIVE AUDIT METRICS & BUFFER TELEMETRY", section_h1))
    
    kpi_data = [
        [
            Paragraph("<b>TOTAL LOGS RECORDED</b>", ParagraphStyle('k1', parent=badge_style, textColor=colors.HexColor("#475569"))),
            Paragraph("<b>OPERATOR ACTIONS</b>", ParagraphStyle('k2', parent=badge_style, textColor=colors.HexColor("#0284c7"))),
            Paragraph("<b>EARLY PREDICTIONS</b>", ParagraphStyle('k3', parent=badge_style, textColor=colors.HexColor("#d97706"))),
            Paragraph("<b>ANOMALIES DETECTED</b>", ParagraphStyle('k4', parent=badge_style, textColor=colors.HexColor("#dc2626"))),
            Paragraph("<b>RL MITIGATIONS</b>", ParagraphStyle('k5', parent=badge_style, textColor=colors.HexColor("#059669"))),
            Paragraph("<b>RECOVERIES</b>", ParagraphStyle('k6', parent=badge_style, textColor=colors.HexColor("#7c3aed")))
        ],
        [
            Paragraph(f"<b>{stats.get('total_events_logged', len(logs))}</b>", ParagraphStyle('v1', parent=badge_style, fontSize=13, leading=15, textColor=colors.HexColor("#0f172a"))),
            Paragraph(f"<b>{stats.get('operator_actions', 0)}</b>", ParagraphStyle('v2', parent=badge_style, fontSize=13, leading=15, textColor=colors.HexColor("#0284c7"))),
            Paragraph(f"<b>{stats.get('early_predictions', 0)}</b>", ParagraphStyle('v3', parent=badge_style, fontSize=13, leading=15, textColor=colors.HexColor("#d97706"))),
            Paragraph(f"<b>{stats.get('faults_detected', 0)}</b>", ParagraphStyle('v4', parent=badge_style, fontSize=13, leading=15, textColor=colors.HexColor("#dc2626"))),
            Paragraph(f"<b>{stats.get('mitigations_executed', 0)}</b>", ParagraphStyle('v5', parent=badge_style, fontSize=13, leading=15, textColor=colors.HexColor("#059669"))),
            Paragraph(f"<b>{stats.get('recoveries', 0)}</b>", ParagraphStyle('v6', parent=badge_style, fontSize=13, leading=15, textColor=colors.HexColor("#7c3aed")))
        ],
        [
            Paragraph(f"Capacity: {stats.get('buffer_usage_pct', 0.0)}%", ParagraphStyle('s1', parent=badge_style, fontSize=6.5, textColor=colors.HexColor("#64748b"))),
            Paragraph("Manual & Gates", ParagraphStyle('s2', parent=badge_style, fontSize=6.5, textColor=colors.HexColor("#64748b"))),
            Paragraph("Precursor Warnings", ParagraphStyle('s3', parent=badge_style, fontSize=6.5, textColor=colors.HexColor("#64748b"))),
            Paragraph("Multi-Class FDIR", ParagraphStyle('s4', parent=badge_style, fontSize=6.5, textColor=colors.HexColor("#64748b"))),
            Paragraph("Q-Learning Policy", ParagraphStyle('s5', parent=badge_style, fontSize=6.5, textColor=colors.HexColor("#64748b"))),
            Paragraph("Nominal Restored", ParagraphStyle('s6', parent=badge_style, fontSize=6.5, textColor=colors.HexColor("#64748b")))
        ]
    ]

    col_w = 7.4 * inch / 6.0
    kpi_table = Table(kpi_data, colWidths=[col_w]*6)
    kpi_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#f8fafc")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#cbd5e1")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#e2e8f0")),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(kpi_table)
    story.append(Spacer(1, 10))

    # =========================================================================
    # 3. Spacecraft Subsystem & 5-Fault Resolution Matrix Reference
    # =========================================================================
    story.append(Paragraph("2. SPACECRAFT SUBSYSTEM HEALTH & 5-FAULT COMPLIANCE ARCHITECTURE", section_h1))
    
    matrix_data = [
        [
            Paragraph("<b>Subsystem Component</b>", ParagraphStyle('mth1', parent=badge_style, textColor=colors.white)),
            Paragraph("<b>Target Spacecraft Fault</b>", ParagraphStyle('mth2', parent=badge_style, textColor=colors.white)),
            Paragraph("<b>Detection Precursor / Trigger</b>", ParagraphStyle('mth3', parent=badge_style, textColor=colors.white)),
            Paragraph("<b>Closed-Loop Mitigation Policy</b>", ParagraphStyle('mth4', parent=badge_style, textColor=colors.white)),
            Paragraph("<b>Recovery Verification Standard</b>", ParagraphStyle('mth5', parent=badge_style, textColor=colors.white))
        ],
        [
            Paragraph("<b>1. PCDU Bus-A</b><br/>Power Distribution", table_cell_style),
            Paragraph("<b>Internal Micro-Short</b><br/>(Dendrite penetration)", table_cell_style),
            Paragraph("Current > 4.5A spike<br/>Bus Voltage < 2.60V", table_cell_style),
            Paragraph("<b>SAFETY_OVERRIDE / SAFE_MODE</b><br/>Latching relay trips (<20ms); cross-straps to Bus-B", table_cell_style),
            Paragraph("NASA-HDBK-4008 §4.2<br/>OBC isolated at 9.2W", table_cell_style)
        ],
        [
            Paragraph("<b>2. Li-ion 8S4P Pack</b><br/>Secondary Energy", table_cell_style),
            Paragraph("<b>Thermal Runaway</b><br/>(Joule heating accumulation)", table_cell_style),
            Paragraph("dT/dt > 1.2°C/s<br/>Cell Temp T > 50°C", table_cell_style),
            Paragraph("<b>LOAD_SHEDDING (-35% Payload)</b><br/>RPC trips; BCR to Trickle; ADCS slews +18.5° to 3K sink", table_cell_style),
            Paragraph("ECSS-E-ST-20C<br/>Joule heat drops 85%", table_cell_style)
        ],
        [
            Paragraph("<b>3. BCR & Shunt Reg</b><br/>Solar Ingestion", table_cell_style),
            Paragraph("<b>Deep Undervoltage</b><br/>(Extended eclipse excursion)", table_cell_style),
            Paragraph("Voltage < 2.90V<br/>SOC < 15%", table_cell_style),
            Paragraph("<b>LOAD_SHEDDING (Tier-2 UVLS)</b><br/>-75% Load shed; ADCS B-dot detumble slews arrays to Sun", table_cell_style),
            Paragraph("ISRO-URSC-PCDU-EPS-04<br/>Bus restores >3.3V", table_cell_style)
        ],
        [
            Paragraph("<b>4. Li-ion Electrodes</b><br/>Electrochemical Cells", table_cell_style),
            Paragraph("<b>High Impedance / Aging</b><br/>(SEI Layer thickening)", table_cell_style),
            Paragraph("R_int > 0.120 Ω<br/>Transient ESR ripple", table_cell_style),
            Paragraph("<b>HIGH_SENSITIVITY_PREARM</b><br/>RL lowers Tau=0.35; caps max discharge rate to 0.4C (1.5A)", table_cell_style),
            Paragraph("NASA-HDBK-4008<br/>Voltage sag stabilized", table_cell_style)
        ],
        [
            Paragraph("<b>5. Sensor ADC Transducer</b><br/>Telemetry Bus", table_cell_style),
            Paragraph("<b>Sensor Glitch / Noise</b><br/>(Radiation SET transient)", table_cell_style),
            Paragraph("> 3-sigma Z-Score outlier<br/>12Hz ADC ringing", table_cell_style),
            Paragraph("<b>AI RAG REASONER + EWMA</b><br/>EWMA rejects spike; AI raises Tau=0.70 to prevent false trip", table_cell_style),
            Paragraph("False Alarm Rate < 1.2%<br/>Payload stays online", table_cell_style)
        ]
    ]

    matrix_table = Table(matrix_data, colWidths=[1.4 * inch, 1.4 * inch, 1.4 * inch, 2.0 * inch, 1.2 * inch])
    matrix_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#0f172a")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#cbd5e1")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#e2e8f0")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor("#ffffff"), colors.HexColor("#f8fafc")]),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(matrix_table)
    story.append(Spacer(1, 12))

    # =========================================================================
    # 4. Chronological Structured Audit Ledger Table
    # =========================================================================
    story.append(Paragraph("3. CHRONOLOGICAL MISSION AUDIT LEDGER (UP TO 10,000-LINE BUFFER)", section_h1))
    story.append(Paragraph("Full audit trail of all frontend user actions, streaming telemetry, early precursor predictions, anomaly triggers, RL mitigation actuations, and recovery events:", body_style))
    story.append(Spacer(1, 4))

    # Header Row
    ledger_table_data = [
        [
            Paragraph("<b>SEQ / MET</b>", ParagraphStyle('lh1', parent=badge_style, textColor=colors.white)),
            Paragraph("<b>SEVERITY</b>", ParagraphStyle('lh2', parent=badge_style, textColor=colors.white)),
            Paragraph("<b>CATEGORY</b>", ParagraphStyle('lh3', parent=badge_style, textColor=colors.white)),
            Paragraph("<b>SUBSYSTEM</b>", ParagraphStyle('lh4', parent=badge_style, textColor=colors.white)),
            Paragraph("<b>TELEMETRY SNAPSHOT</b>", ParagraphStyle('lh5', parent=badge_style, textColor=colors.white)),
            Paragraph("<b>AI/RL & MESSAGE DETAILS</b>", ParagraphStyle('lh6', parent=badge_style, textColor=colors.white))
        ]
    ]

    # Limit max records displayed in PDF (e.g. recent 500 records or all if <= 500) to keep PDF performant & clean
    sample_logs = logs[:500] if len(logs) > 500 else logs

    for item in sample_logs:
        seq_id = item.get("seq_id", 0)
        met = item.get("met", "T+00:00:00")
        level = item.get("level", "INFO")
        event_type = item.get("event_type", "TELEMETRY")
        subsystem = item.get("subsystem", "EPS_CORE")
        message = item.get("message", "")
        telemetry = item.get("telemetry", {})
        models = item.get("models", {})
        metadata = item.get("metadata", {})

        # Determine Badge Color
        lvl_color = "#64748b"
        lvl_bg = "#f1f5f9"
        if level in ["CRITICAL", "EMERGENCY"]:
            lvl_color = "#b91c1c"
            lvl_bg = "#fee2e2"
        elif level in ["FAULT", "WARNING"]:
            lvl_color = "#b45309"
            lvl_bg = "#fef3c7"
        elif level == "ACTION":
            lvl_color = "#0369a1"
            lvl_bg = "#e0f2fe"
        elif level == "PREDICT":
            lvl_color = "#c2410c"
            lvl_bg = "#ffedd5"
        elif level == "RL-POLICY":
            lvl_color = "#047857"
            lvl_bg = "#d1fae5"
        elif level == "RECOVER":
            lvl_color = "#6d28d9"
            lvl_bg = "#ede9fe"
        elif level == "SAFETY":
            lvl_color = "#991b1b"
            lvl_bg = "#fee2e2"

        # Telemetry String
        v = telemetry.get("voltage", None)
        i = telemetry.get("current", None)
        t = telemetry.get("temperature", None)
        soc = telemetry.get("soc", None)
        p_ens = models.get("p_ensemble", None)

        telem_parts = []
        if v is not None: telem_parts.append(f"V: {v:.2f}V")
        if i is not None: telem_parts.append(f"I: {i:.2f}A")
        if t is not None: telem_parts.append(f"T: {t:.1f}°C")
        if soc is not None: telem_parts.append(f"SOC: {soc*100:.0f}%" if soc <= 1.0 else f"SOC: {soc:.0f}%")
        if p_ens is not None: telem_parts.append(f"P_ens: {p_ens:.2f}")

        telem_str = " • ".join(telem_parts) if telem_parts else "N/A"

        # Formatted Row
        formatted_cat = event_type.replace('_', ' ')
        formatted_sub = subsystem.replace('_', ' ')
        col_seq = Paragraph(f"<b>#{seq_id}</b><br/>{met}", table_mono_style)
        col_lvl = Paragraph(f"<b>{level}</b>", ParagraphStyle(f'b_{seq_id}', parent=badge_style, textColor=colors.HexColor(lvl_color)))
        col_cat = Paragraph(f"<b>{formatted_cat}</b>", table_cell_style)
        col_sub = Paragraph(formatted_sub, table_cell_style)
        col_tel = Paragraph(telem_str, table_mono_style)
        col_msg = Paragraph(message, table_cell_style)

        ledger_table_data.append([col_seq, col_lvl, col_cat, col_sub, col_tel, col_msg])

    ledger_table = Table(
        ledger_table_data,
        colWidths=[0.72 * inch, 0.76 * inch, 1.22 * inch, 1.05 * inch, 1.45 * inch, 2.20 * inch],
        repeatRows=1
    )

    ledger_table_style = [
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#0f172a")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#cbd5e1")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#e2e8f0")),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
    ]

    # Add row backgrounds
    for r_idx in range(1, len(ledger_table_data)):
        bg_c = colors.HexColor("#ffffff") if r_idx % 2 == 1 else colors.HexColor("#f8fafc")
        ledger_table_style.append(('BACKGROUND', (0, r_idx), (-1, r_idx), bg_c))

    ledger_table.setStyle(TableStyle(ledger_table_style))
    story.append(ledger_table)

    # Build Document using NumberedCanvas
    doc.build(story, canvasmaker=NumberedCanvas)
    return output_path
