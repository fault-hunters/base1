import * as React from "react"
import { NavLink } from "react-router-dom"
//import { addPropertyControls, ControlType, useRouter } from "framer"

// --- Types ---
type DetectResult = {
    verdict: "OK" | "NG"
    score: number
    defects: string[]
    message: string
}

type HistoryItem = {
    id: string
    projectName: string
    timestamp: string
    refUrl: string // 원본 이미지
    tarUrl: string // 생성 이미지
    name: string
    result: DetectResult
}

const HISTORY_KEY = "rawer:history_v1"
const FONT_STACK = '"Switzer", "Noto Sans KR", Inter, system-ui, sans-serif'

const linkStyle = ({ isActive }: { isActive: boolean }) => ({
    padding: "6px 10px",
    borderRadius: 8,
    fontSize: 11,
    fontWeight: 700,
    textDecoration: "none",
    color: isActive ? "#000" : "#666",
    background: isActive ? "#F2F2F2" : "transparent",
})

function safeJsonParse<T>(s: string | null): T | null {
    if (!s) return null
    try {
        return JSON.parse(s) as T
    } catch {
        return null
    }
}

// --- UI Atoms: 시각화 차트 ---
const DonutChart = ({
    ok,
    total,
    size = 100,
}: {
    ok: number
    total: number
    size?: number
}) => {
    const percentage = total > 0 ? (ok / total) * 100 : 0
    const strokeDasharray = `${percentage} ${100 - percentage}`
    return (
        <div style={{ position: "relative", width: size, height: size }}>
            <svg viewBox="0 0 36 36" style={{ transform: "rotate(-90deg)" }}>
                <circle
                    cx="18"
                    cy="18"
                    r="15.9"
                    fill="transparent"
                    stroke="#F0F0F0"
                    strokeWidth="3"
                />
                <circle
                    cx="18"
                    cy="18"
                    r="15.9"
                    fill="transparent"
                    stroke="#4CD964"
                    strokeWidth="3"
                    strokeDasharray={strokeDasharray}
                    strokeDashoffset="0"
                    strokeLinecap="round"
                />
            </svg>
            <div
                style={{
                    position: "absolute",
                    top: "50%",
                    left: "50%",
                    transform: "translate(-50%, -50%)",
                    fontSize: 14,
                    fontWeight: 800,
                }}
            >
                {Math.round(percentage)}%
            </div>
        </div>
    )
}

const ScoreBar = ({ score }: { score: number }) => (
    <div
        style={{
            width: "100%",
            height: 4,
            background: "#F0F0F0",
            borderRadius: 2,
            overflow: "hidden",
            marginTop: 4,
        }}
    >
        <div
            style={{
                width: `${score}%`,
                height: "100%",
                background: score > 70 ? "#4CD964" : "#FF3B30",
                transition: "width 0.5s ease",
            }}
        />
    </div>
)

export default function HistoryDashboard() {
    const [history, setHistory] = React.useState<HistoryItem[]>([])
    const [selectedProject, setSelectedProject] =
        React.useState<string>("All Projects")
    const [statusFilter, setStatusFilter] = React.useState<"All" | "OK" | "NG">(
        "All"
    )
    const [selectedIdx, setSelectedIdx] = React.useState<number>(0)

    const refreshHistory = React.useCallback(() => {
        const stored = safeJsonParse<HistoryItem[]>(
            localStorage.getItem(HISTORY_KEY)
        )
        if (stored) {
            const sorted = [...stored].sort(
                (a, b) =>
                    new Date(b.timestamp).getTime() -
                    new Date(a.timestamp).getTime()
            )
            setHistory(sorted)
        }
    }, [])

    React.useEffect(() => {
        refreshHistory()
        window.addEventListener("focus", refreshHistory)
        return () => window.removeEventListener("focus", refreshHistory)
    }, [refreshHistory])

    // 프로젝트 리스트 추출 (projectName이 폴더 이름으로 들어갑니다)
    const projectList = React.useMemo(() => {
        const names = history.map((h) => h.projectName || "Default Project")
        return ["All Projects", ...Array.from(new Set(names))]
    }, [history])

    const filteredHistory = React.useMemo(() => {
        return history.filter((h) => {
            const hProjectName = h.projectName || "Default Project"
            const matchProject =
                selectedProject === "All Projects" ||
                hProjectName === selectedProject
            const matchStatus =
                statusFilter === "All" || h.result.verdict === statusFilter
            return matchProject && matchStatus
        })
    }, [history, selectedProject, statusFilter])

    const stats = React.useMemo(() => {
        const total = filteredHistory.length
        const ok = filteredHistory.filter(
            (h) => h.result.verdict === "OK"
        ).length
        const ng = total - ok
        const avgScore =
            total > 0
                ? Math.round(
                      filteredHistory.reduce(
                          (acc, cur) => acc + cur.result.score,
                          0
                      ) / total
                  )
                : 0
        return { total, ok, ng, avgScore }
    }, [filteredHistory])

    const selectedItem = filteredHistory[selectedIdx]

    const Card = ({ title, children, style }: any) => (
        <div
            style={{
                border: "1px solid rgba(0,0,0,0.08)",
                borderRadius: 16,
                padding: 16,
                background: "white",
                display: "flex",
                flexDirection: "column",
                ...style,
            }}
        >
            <div
                style={{
                    fontWeight: 600,
                    fontSize: 11,
                    color: "#AAA",
                    marginBottom: 12,
                    textTransform: "uppercase",
                }}
            >
                {title}
            </div>
            {children}
        </div>
    )

    return (
        <div
            style={{
                width: "100%",
                height: "100vh",
                background: "#FAFAFA",
                fontFamily: FONT_STACK,
                display: "flex",
                flexDirection: "column",
                overflow: "hidden",
            }}
        >
            <div
                style={{
                    padding: "16px 24px",
                    display: "grid",
                    gridTemplateColumns: "1fr auto 1fr",
                    alignItems: "center",
                    background: "#FFFFFF",
                    borderBottom: "1px solid #F0F0F0",
                }}
            >
                <div
                    style={{
                        fontSize: 24,
                        fontWeight: 800,
                        justifySelf: "start",
                    }}
                >
                    History
                </div>
                <div style={{ display: "flex", gap: 8, justifySelf: "center" }}>
                    <NavLink to="/" style={linkStyle} end>
                        AIGC Generator
                    </NavLink>
                    <NavLink to="/detect" style={linkStyle}>
                        Detection
                    </NavLink>
                    <NavLink to="/history" style={linkStyle}>
                        History
                    </NavLink>
                </div>
                <div />
            </div>
            <div
                style={{
                    flex: 1,
                    display: "flex",
                    overflow: "hidden",
                }}
            >
            {/* --- Left Sidebar: Project Folders --- */}
            <div
                style={{
                    width: 260,
                    background: "#FFF",
                    borderRight: "1px solid #EEE",
                    display: "flex",
                    flexDirection: "column",
                    flexShrink: 0,
                }}
            >
                <div
                    style={{
                        padding: "24px 20px",
                        fontSize: 20,
                        fontWeight: 800,
                    }}
                >
                    Projects
                </div>
                <div style={{ flex: 1, overflowY: "auto", padding: "0 10px" }}>
                    {projectList.map((name) => (
                        <div
                            key={name}
                            onClick={() => {
                                setSelectedProject(name)
                                setSelectedIdx(0)
                            }}
                            style={{
                                padding: "12px 16px",
                                borderRadius: 10,
                                cursor: "pointer",
                                fontSize: 13,
                                fontWeight: 600,
                                marginBottom: 4,
                                background:
                                    selectedProject === name
                                        ? "#F0F0F0"
                                        : "transparent",
                                color:
                                    selectedProject === name ? "#000" : "#666",
                                display: "flex",
                                justifyContent: "space-between",
                            }}
                        >
                            <span>
                                {name === "All Projects" ? "📁 " : "📄 "} {name}
                            </span>
                        </div>
                    ))}
                </div>
            </div>

            {/* --- Main Dashboard Area --- */}
            <div
                style={{
                    flex: 1,
                    display: "flex",
                    flexDirection: "column",
                    minWidth: 0,
                }}
            >
                <div
                    style={{
                        padding: "16px 24px",
                        display: "flex",
                        justifyContent: "space-between",
                        alignItems: "center",
                        background: "#FFF",
                        borderBottom: "1px solid #EEE",
                    }}
                >
                    <div style={{ fontSize: 20, fontWeight: 700 }}>
                        {selectedProject}{" "}
                        <span style={{ color: "#AAA", fontWeight: 400 }}>
                            History
                        </span>
                    </div>
                    <button
                        onClick={() =>
                            confirm("히스토리를 삭제하시겠습니까?") &&
                            (localStorage.setItem(HISTORY_KEY, "[]"),
                            setHistory([]))
                        }
                        style={{
                            padding: "8px 16px",
                            borderRadius: 8,
                            border: "1px solid #EEE",
                            background: "#FFF",
                            cursor: "pointer",
                            fontSize: 12,
                        }}
                    >
                        Clear History
                    </button>
                </div>

                {/* Filters */}
                <div
                    style={{
                        padding: "12px 24px",
                        display: "flex",
                        gap: 10,
                        background: "#FFF",
                        borderBottom: "1px solid #F5F5F5",
                    }}
                >
                    {(["All", "OK", "NG"] as const).map((s) => (
                        <button
                            key={s}
                            onClick={() => {
                                setStatusFilter(s)
                                setSelectedIdx(0)
                            }}
                            style={{
                                padding: "6px 14px",
                                borderRadius: 6,
                                fontSize: 12,
                                fontWeight: 700,
                                cursor: "pointer",
                                border: "none",
                                background:
                                    statusFilter === s ? "#000" : "#F5F5F5",
                                color: statusFilter === s ? "#FFF" : "#666",
                            }}
                        >
                            {s}
                        </button>
                    ))}
                </div>

                <div
                    style={{
                        flex: 1,
                        display: "flex",
                        padding: 20,
                        gap: 20,
                        overflow: "hidden",
                    }}
                >
                    {/* Log List */}
                    <div
                        style={{
                            width: 340,
                            display: "flex",
                            flexDirection: "column",
                            gap: 12,
                            overflowY: "auto",
                            paddingRight: 4,
                        }}
                    >
                        {filteredHistory.length === 0 ? (
                            <div
                                style={{
                                    textAlign: "center",
                                    color: "#CCC",
                                    marginTop: 40,
                                }}
                            >
                                기록이 없습니다.
                            </div>
                        ) : (
                            filteredHistory.map((item, idx) => (
                                <div
                                    key={item.id}
                                    onClick={() => setSelectedIdx(idx)}
                                    style={{
                                        padding: 14,
                                        borderRadius: 14,
                                        border:
                                            selectedIdx === idx
                                                ? "1px solid #000"
                                                : "1px solid #EEE",
                                        background:
                                            selectedIdx === idx
                                                ? "#FFF"
                                                : "transparent",
                                        cursor: "pointer",
                                        transition: "all 0.2s",
                                    }}
                                >
                                    <div
                                        style={{
                                            display: "flex",
                                            justifyContent: "space-between",
                                            marginBottom: 8,
                                            fontSize: 10,
                                            fontWeight: 700,
                                        }}
                                    >
                                        <span style={{ color: "#AAA" }}>
                                            {new Date(
                                                item.timestamp
                                            ).toLocaleTimeString()}
                                        </span>
                                        <span
                                            style={{
                                                color:
                                                    item.result.verdict === "OK"
                                                        ? "#4CD964"
                                                        : "#FF3B30",
                                            }}
                                        >
                                            {item.result.verdict}
                                        </span>
                                    </div>
                                    <div
                                        style={{
                                            display: "flex",
                                            gap: 10,
                                            alignItems: "center",
                                        }}
                                    >
                                        <img
                                            src={item.tarUrl}
                                            style={{
                                                width: 40,
                                                height: 40,
                                                borderRadius: 8,
                                                objectFit: "cover",
                                            }}
                                        />
                                        <div
                                            style={{
                                                flex: 1,
                                                overflow: "hidden",
                                            }}
                                        >
                                            <div
                                                style={{
                                                    fontSize: 13,
                                                    fontWeight: 700,
                                                    whiteSpace: "nowrap",
                                                    overflow: "hidden",
                                                    textOverflow: "ellipsis",
                                                }}
                                            >
                                                {item.name}
                                            </div>
                                            <ScoreBar
                                                score={item.result.score}
                                            />
                                        </div>
                                    </div>
                                </div>
                            ))
                        )}
                    </div>

                    {/* Detail View */}
                    <div
                        style={{
                            flex: 1,
                            display: "flex",
                            flexDirection: "column",
                            gap: 20,
                            overflowY: "auto",
                        }}
                    >
                        <div
                            style={{
                                display: "grid",
                                gridTemplateColumns: "1.5fr 1fr 1fr 1fr",
                                gap: 12,
                            }}
                        >
                            <Card title="Pass Rate Analysis">
                                <div
                                    style={{
                                        display: "flex",
                                        alignItems: "center",
                                        gap: 16,
                                    }}
                                >
                                    <DonutChart
                                        ok={stats.ok}
                                        total={stats.total}
                                    />
                                    <div>
                                        <div
                                            style={{
                                                fontSize: 20,
                                                fontWeight: 800,
                                            }}
                                        >
                                            {stats.ok} / {stats.total}
                                        </div>
                                        <div
                                            style={{
                                                fontSize: 11,
                                                color: "#AAA",
                                            }}
                                        >
                                            합격 / 전체 분석 수
                                        </div>
                                    </div>
                                </div>
                            </Card>
                            <Card title="Avg Score">
                                <div style={{ fontSize: 24, fontWeight: 800 }}>
                                    {stats.avgScore}{" "}
                                    <span style={{ fontSize: 14 }}>pt</span>
                                </div>
                            </Card>
                            <Card title="OK" style={{ color: "#4CD964" }}>
                                <div style={{ fontSize: 24, fontWeight: 800 }}>
                                    {stats.ok}
                                </div>
                            </Card>
                            <Card title="NG" style={{ color: "#FF3B30" }}>
                                <div style={{ fontSize: 24, fontWeight: 800 }}>
                                    {stats.ng}
                                </div>
                            </Card>
                        </div>

                        {selectedItem ? (
                            <div style={{ display: "flex", gap: 16, flex: 1 }}>
                                <Card
                                    title="Comparison (Ref vs Tar)"
                                    style={{
                                        flex: 2,
                                        flexDirection: "row",
                                        gap: 12,
                                        alignItems: "center",
                                    }}
                                >
                                    <div
                                        style={{ flex: 1, textAlign: "center" }}
                                    >
                                        <div
                                            style={{
                                                fontSize: 10,
                                                color: "#DDD",
                                                marginBottom: 6,
                                                fontWeight: 800,
                                            }}
                                        >
                                            REFERENCE (ORIGINAL)
                                        </div>
                                        {/* ✅ 수정: 원본 이미지(refUrl)가 있으면 노출, 없으면 메시지 노출 */}
                                        {selectedItem.refUrl ? (
                                            <img
                                                src={selectedItem.refUrl}
                                                style={{
                                                    width: "100%",
                                                    borderRadius: 10,
                                                    border: "1px solid #F0F0F0",
                                                    aspectRatio: "1/1",
                                                    objectFit: "cover",
                                                }}
                                            />
                                        ) : (
                                            <div
                                                style={{
                                                    width: "100%",
                                                    aspectRatio: "1/1",
                                                    borderRadius: 10,
                                                    border: "1px solid #F0F0F0",
                                                    background: "#F9F9F9",
                                                    display: "flex",
                                                    alignItems: "center",
                                                    justifyContent: "center",
                                                    fontSize: 12,
                                                    color: "#FF3B30",
                                                    fontWeight: 600,
                                                }}
                                            >
                                                ref IMG가 없습니다
                                            </div>
                                        )}
                                    </div>
                                    <div
                                        style={{ flex: 1, textAlign: "center" }}
                                    >
                                        <div
                                            style={{
                                                fontSize: 10,
                                                color: "#DDD",
                                                marginBottom: 6,
                                                fontWeight: 800,
                                            }}
                                        >
                                            TARGET (GENERATED)
                                        </div>
                                        <img
                                            src={selectedItem.tarUrl}
                                            style={{
                                                width: "100%",
                                                borderRadius: 10,
                                                border: "1px solid #F0F0F0",
                                                aspectRatio: "1/1",
                                                objectFit: "cover",
                                            }}
                                        />
                                    </div>
                                </Card>
                                <Card
                                    title="Diagnosis Detail"
                                    style={{ flex: 1 }}
                                >
                                    <div
                                        style={{
                                            fontSize: 40,
                                            fontWeight: 900,
                                            color:
                                                selectedItem.result.verdict ===
                                                "OK"
                                                    ? "#4CD964"
                                                    : "#FF3B30",
                                            marginBottom: 12,
                                        }}
                                    >
                                        {selectedItem.result.verdict}
                                    </div>
                                    <div
                                        style={{
                                            fontSize: 13,
                                            color: "#555",
                                            lineHeight: 1.6,
                                            marginBottom: 16,
                                        }}
                                    >
                                        {selectedItem.result.message}
                                    </div>
                                    <div
                                        style={{
                                            display: "flex",
                                            flexWrap: "wrap",
                                            gap: 6,
                                        }}
                                    >
                                        {selectedItem.result.defects.map(
                                            (d) => (
                                                <span
                                                    key={d}
                                                    style={{
                                                        fontSize: 10,
                                                        fontWeight: 800,
                                                        padding: "4px 8px",
                                                        background: "#FFF0F0",
                                                        color: "#FF3B30",
                                                        borderRadius: 4,
                                                    }}
                                                >
                                                    {d}
                                                </span>
                                            )
                                        )}
                                    </div>
                                </Card>
                            </div>
                        ) : (
                            <div
                                style={{
                                    textAlign: "center",
                                    color: "#BBB",
                                    marginTop: 80,
                                }}
                            >
                                로그를 선택하여 상세 데이터를 확인하세요.
                            </div>
                        )}
                    </div>
                </div>
            </div>
            </div>
        </div>
    )
}
