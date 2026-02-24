import * as React from "react"
import { NavLink } from "react-router-dom"
//import { addPropertyControls, ControlType, useRouter } from "framer"

// --- Types ---
type DetectResult = {
    verdict: "OK" | "NG"
    score: number
    defects: string[]
    message: string
    boxes: any[]
}
type TarImage = { id: string; url: string; file?: File; result?: DetectResult }
type RefGroup = {
    id: string
    refUrl: string
    refFile?: File
    name: string
    tarImages: TarImage[]
}

const GEN_KEY = "rawer:last_generation_v1"
const FONT_STACK = '"Switzer", "Noto Sans KR", Inter, system-ui, sans-serif'
const HISTORY_KEY = "rawer:history_v1"
const PROJECT_STORAGE_KEY = "projectName"

function safeJsonParse<T>(s: string | null): T | null {
    if (!s) return null
    try {
        return JSON.parse(s) as T
    } catch {
        return null
    }
}

export default function Detection() {
    //const router = useRouter()

    // Refs
    const newTarInputRef = React.useRef<HTMLInputElement>(null)

    // States
    const [targetGroupId, setTargetGroupId] = React.useState<string | null>(
        null
    )
    const [groups, setGroups] = React.useState<RefGroup[]>([])
    const [activePair, setActivePair] = React.useState<{
        gIdx: number
        tIdx: number
    }>({ gIdx: -1, tIdx: -1 })
    const [running, setRunning] = React.useState(false)

    // 통계 계산
    const stats = React.useMemo(() => {
        let total = 0,
            ok = 0,
            ng = 0
        const ngList: string[] = []
        groups.forEach((g) => {
            g.tarImages.forEach((t) => {
                if (t.result) {
                    total++
                    if (t.result.verdict === "OK") ok++
                    else {
                        ng++
                        ngList.push(t.url)
                    }
                }
            })
        })
        return {
            total,
            ok,
            ng,
            rate: total > 0 ? Math.round((ok / total) * 100) : 0,
            ngList,
        }
    }, [groups])

    // 초기 데이터 로드
    React.useEffect(() => {
        const stored = safeJsonParse<{
            images: any[]
            referenceImageUrls?: string[]
            selectedIndex?: number
        }>(localStorage.getItem(GEN_KEY))

        if (stored?.images?.length) {
            setGroups([
                {
                    id: "gen-group",
                    name: "Latest Generation",
                    refUrl: stored.referenceImageUrls?.[0] || "",
                    tarImages: stored.images.map((img, i) => ({
                        id: `tar-${i}`,
                        url: img.url,
                    })),
                },
            ])
            setActivePair({ gIdx: 0, tIdx: stored.selectedIndex || 0 })
        }
    }, [])

    // 핸들러: 새 그룹 추가 (Ref 이미지 선택)
    const handleAddGroup = (e: React.ChangeEvent<HTMLInputElement>) => {
        const files = Array.from(e.target.files || [])
        const newGroups = files.map((file) => ({
            id: `manual-group-${Date.now()}-${Math.random()}`,
            name: file.name.split(".")[0],
            refUrl: URL.createObjectURL(file),
            refFile: file,
            tarImages: [],
        }))
        setGroups([...groups, ...newGroups])
        e.target.value = ""
    }

    // 핸들러: 특정 그룹에 Target 이미지 추가
    const handleAddTar = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (!targetGroupId) return
        const files = Array.from(e.target.files || [])
        const newTars = files.map((file) => ({
            id: `tar-${Date.now()}-${Math.random()}`,
            url: URL.createObjectURL(file),
            file,
        }))
        setGroups((prev) =>
            prev.map((g) =>
                g.id === targetGroupId
                    ? { ...g, tarImages: [...g.tarImages, ...newTars] }
                    : g
            )
        )
        setTargetGroupId(null)
        e.target.value = ""
    }

    const exportNGData = async () => {
        if (stats.ng === 0) return alert("전송할 NG 데이터가 없습니다.")
        const payload = {
            timestamp: new Date().toISOString(),
            stats: {
                total: stats.total,
                ng_count: stats.ng,
                pass_rate: stats.rate,
            },
            ng_assets: stats.ngList,
        }
        try {
            const response = await fetch("http://127.0.0.1:8000/api/ng-data", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            })
            if (!response.ok) throw new Error("전송 실패")
            alert("NG 데이터가 전송되었습니다.")
        } catch (e) {
            alert(e)
            //alert("서버 연결을 확인하세요.")
        }
    }

    const runAllInspection = async () => {
        if (groups.length === 0) return
        setRunning(true)
        const currentProjectName =
            localStorage.getItem(PROJECT_STORAGE_KEY) || "Default Project"

        try {
            const updated = await Promise.all(
                groups.map(async (g) => {
                    const tars = await Promise.all(
                        g.tarImages.map(async (t) => {
                            const formData = new FormData()
                            
                            if (g.refFile) {
                            formData.append("ref_file", g.refFile)
                            } else {
                            // 예전 blob/url만 있는 경우 대비
                            const refBlob = await fetch(g.refUrl).then(r => r.blob())
                            formData.append("ref_file", refBlob, "ref.png")
                            }

                            if (t.file) {
                            formData.append("tar_file", t.file)
                            } else {
                            const tarBlob = await fetch(t.url).then(r => r.blob())
                            formData.append("tar_file", tarBlob, "tar.png")
                            }

                            formData.append("project_name", currentProjectName)

                            const response = await fetch(
                                "http://127.0.0.1:8000/api/detect",
                                {
                                    method: "POST",
                                    body: formData,
                                }
                            )
                            if (!response.ok) {
                                const text = await response.text()
                                throw new Error(text || "server_error")
                            }
                            const result = await response.json()

                            const newHistoryItem = {
                                id: `hist-${Date.now()}-${Math.random()}`,
                                projectName: currentProjectName,
                                timestamp: new Date().toISOString(),
                                refUrl: g.refUrl,
                                tarUrl: t.url,
                                result: result,
                            }
                            const existing = JSON.parse(
                                localStorage.getItem(HISTORY_KEY) || "[]"
                            )
                            localStorage.setItem(
                                HISTORY_KEY,
                                JSON.stringify([newHistoryItem, ...existing])
                            )

                            return { ...t, result }
                        })
                    )
                    return { ...g, tarImages: tars }
                })
            )
            setGroups(updated)
        } catch (e) {
            // alert("서버 에러")
            alert(e)
        } finally {
            setRunning(false)
        }
    }

    const Card = ({ title, children, style }: any) => (
        <div
            style={{
                border: "1px solid rgba(0,0,0,0.1)",
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
                    fontSize: 13,
                    color: "#999",
                    marginBottom: 12,
                    textTransform: "uppercase",
                }}
            >
                {title}
            </div>
            {children}
        </div>
    )

    const selectedTar = groups[activePair.gIdx]?.tarImages[activePair.tIdx]

    return (
        <div
            style={{
                width: "100%",
                height: "100vh",
                background: "#FFFFFF",
                fontFamily: FONT_STACK,
                display: "flex",
                flexDirection: "column",
                overflow: "hidden",
            }}
        >
            {/* Header */}
            <div
                style={{
                    padding: "16px 24px",
                    display: "grid",
                    gridTemplateColumns: "1fr auto 1fr",
                    alignItems: "center",
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
                    Fault Detection
                </div>
                <div
                    style={{
                        display: "flex",
                        gap: 8,
                        justifySelf: "center",
                    }}
                >
                    <NavLink
                        to="/"
                        style={({ isActive }) => ({
                            padding: "6px 10px",
                            borderRadius: 8,
                            fontSize: 11,
                            fontWeight: 700,
                            textDecoration: "none",
                            color: isActive ? "#000" : "#666",
                            background: isActive ? "#F2F2F2" : "transparent",
                        })}
                        end
                    >
                        AIGC Generator
                    </NavLink>
                    <NavLink
                        to="/detect"
                        style={({ isActive }) => ({
                            padding: "6px 10px",
                            borderRadius: 8,
                            fontSize: 11,
                            fontWeight: 700,
                            textDecoration: "none",
                            color: isActive ? "#000" : "#666",
                            background: isActive ? "#F2F2F2" : "transparent",
                        })}
                    >
                        Detection
                    </NavLink>
                    <NavLink
                        to="/history"
                        style={({ isActive }) => ({
                            padding: "6px 10px",
                            borderRadius: 8,
                            fontSize: 11,
                            fontWeight: 700,
                            textDecoration: "none",
                            color: isActive ? "#000" : "#666",
                            background: isActive ? "#F2F2F2" : "transparent",
                        })}
                    >
                        History
                    </NavLink>
                </div>
                <div
                    style={{
                        display: "flex",
                        gap: 12,
                        justifySelf: "end",
                    }}
                >
                    <button
                        onClick={exportNGData}
                        style={{
                            background: "#FFF",
                            border: "1px solid #EEE",
                            borderRadius: 10,
                            padding: "8px 16px",
                            cursor: "pointer",
                        }}
                    >
                        Export NG Data
                    </button>
                    <button
                        onClick={runAllInspection}
                        disabled={running}
                        style={{
                            background: "#000",
                            color: "#FFF",
                            border: "none",
                            borderRadius: 10,
                            padding: "8px 20px",
                            cursor: "pointer",
                            opacity: running ? 0.5 : 1,
                        }}
                    >
                        {running ? "Analyzing..." : "Run Inspection"}
                    </button>
                </div>
            </div>

            <div
                style={{
                    display: "flex",
                    flex: 1,
                    padding: 20,
                    gap: 20,
                    minHeight: 0,
                }}
            >
                {/* Left: Sidebar */}
                <div
                    style={{
                        width: 400,
                        borderRight: "1px solid #F0F0F0",
                        paddingRight: 20,
                        overflowY: "auto",
                        display: "flex",
                        flexDirection: "column",
                        gap: 16,
                    }}
                >
                    <div
                        style={{
                            display: "flex",
                            justifyContent: "space-between",
                            alignItems: "center",
                            fontSize: 14,
                            fontWeight: 700,
                        }}
                    >
                        <span>Ref-Tar Groups</span>
                        <label style={{ color: "#007AFF", cursor: "pointer" }}>
                            + New Group
                            <input
                                type="file"
                                hidden
                                multiple
                                onChange={handleAddGroup}
                            />
                        </label>
                    </div>

                    {groups.map((g, gIdx) => (
                        <div
                            key={g.id}
                            style={{
                                padding: 14,
                                borderRadius: 14,
                                border:
                                    activePair.gIdx === gIdx
                                        ? "1px solid #000"
                                        : "1px solid #EEE",
                                background:
                                    activePair.gIdx === gIdx
                                        ? "#FAFAFA"
                                        : "#FFF",
                            }}
                        >
                            <div
                                style={{
                                    display: "flex",
                                    justifyContent: "space-between",
                                    alignItems: "center",
                                    marginBottom: 12,
                                }}
                            >
                                <div
                                    style={{
                                        display: "flex",
                                        gap: 8,
                                        alignItems: "center",
                                    }}
                                >
                                    <img
                                        src={g.refUrl}
                                        style={{
                                            width: 28,
                                            height: 28,
                                            borderRadius: 6,
                                            objectFit: "cover",
                                            border: "1px solid #EEE",
                                        }}
                                    />
                                    <div
                                        style={{
                                            fontSize: 12,
                                            fontWeight: 700,
                                        }}
                                    >
                                        {g.name}
                                    </div>
                                </div>
                                <button
                                    onClick={() => {
                                        setTargetGroupId(g.id)
                                        newTarInputRef.current?.click()
                                    }}
                                    style={{
                                        fontSize: 11,
                                        color: "#007AFF",
                                        border: "none",
                                        background: "none",
                                        cursor: "pointer",
                                        fontWeight: 600,
                                    }}
                                >
                                    + Add Tar
                                </button>
                            </div>
                            <div
                                style={{
                                    display: "grid",
                                    gridTemplateColumns: "repeat(4, 1fr)",
                                    gap: 6,
                                }}
                            >
                                {g.tarImages.map((t, tIdx) => (
                                    <div
                                        key={t.id}
                                        onClick={() =>
                                            setActivePair({ gIdx, tIdx })
                                        }
                                        style={{
                                            aspectRatio: "1/1",
                                            borderRadius: 8,
                                            border:
                                                activePair.gIdx === gIdx &&
                                                activePair.tIdx === tIdx
                                                    ? "2px solid #000"
                                                    : "1px solid #EEE",
                                            overflow: "hidden",
                                            cursor: "pointer",
                                        }}
                                    >
                                        <img
                                            src={t.url}
                                            style={{
                                                width: "100%",
                                                height: "100%",
                                                objectFit: "cover",
                                            }}
                                        />
                                    </div>
                                ))}
                            </div>
                        </div>
                    ))}
                </div>

                {/* Right: Dashboard */}
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
                            gridTemplateColumns: "repeat(4, 1fr)",
                            gap: 12,
                        }}
                    >
                        <Card
                            title="Pass Rate"
                            style={{ background: "#000", color: "#4AF2A1" }}
                        >
                            <div style={{ fontSize: 28, fontWeight: 800 }}>
                                {stats.rate}%
                            </div>
                        </Card>
                        <Card title="Total Assets">
                            <div style={{ fontSize: 28, fontWeight: 800 }}>
                                {stats.total}
                            </div>
                        </Card>
                        <Card title="OK" style={{ color: "#2E7D32" }}>
                            <div style={{ fontSize: 28, fontWeight: 800 }}>
                                {stats.ok}
                            </div>
                        </Card>
                        <Card title="NG" style={{ color: "#D32F2F" }}>
                            <div style={{ fontSize: 28, fontWeight: 800 }}>
                                {stats.ng}
                            </div>
                        </Card>
                    </div>

                    <div style={{ display: "flex", gap: 16, flex: 1 }}>
                        <Card
                            title="Comparison"
                            style={{
                                flex: 2,
                                flexDirection: "row",
                                gap: 12,
                                alignItems: "center",
                            }}
                        >
                            <div style={{ flex: 1, textAlign: "center" }}>
                                <div
                                    style={{
                                        fontSize: 10,
                                        color: "#BBB",
                                        fontWeight: 700,
                                        marginBottom: 8,
                                    }}
                                >
                                    REFERENCE
                                </div>
                                {groups[activePair.gIdx]?.refUrl ? (
                                    <img
                                        src={groups[activePair.gIdx].refUrl}
                                        style={{
                                            width: "100%",
                                            borderRadius: 10,
                                            border: "1px solid #EEE",
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
                            <div style={{ flex: 1, textAlign: "center" }}>
                                <div
                                    style={{
                                        fontSize: 10,
                                        color: "#BBB",
                                        fontWeight: 700,
                                        marginBottom: 8,
                                    }}
                                >
                                    TARGET
                                </div>
                                {selectedTar && (
                                    <img
                                        src={selectedTar.url}
                                        style={{
                                            width: "100%",
                                            borderRadius: 10,
                                            border: "1px solid #EEE",
                                        }}
                                    />
                                )}
                            </div>
                        </Card>

                        <Card title="Diagnostic Result" style={{ flex: 1 }}>
                            {selectedTar?.result ? (
                                <div
                                    style={{
                                        display: "flex",
                                        flexDirection: "column",
                                        gap: 16,
                                    }}
                                >
                                    <div
                                        style={{
                                            fontSize: 44,
                                            fontWeight: 900,
                                            color:
                                                selectedTar.result.verdict ===
                                                "OK"
                                                    ? "#2E7D32"
                                                    : "#D32F2F",
                                        }}
                                    >
                                        {selectedTar.result.verdict}
                                    </div>
                                    <div
                                        style={{ fontSize: 13, color: "#555" }}
                                    >
                                        {selectedTar.result.message}
                                    </div>
                                    <div
                                        style={{
                                            fontSize: 11,
                                            fontWeight: 700,
                                            color: "#999",
                                        }}
                                    >
                                        QUALITY SCORE:{" "}
                                        {Math.round(selectedTar.result.score)}
                                    </div>
                                </div>
                            ) : (
                                <div
                                    style={{
                                        textAlign: "center",
                                        color: "#BBB",
                                        marginTop: 40,
                                    }}
                                >
                                    Select an image to view analysis.
                                </div>
                            )}
                        </Card>
                    </div>
                </div>
            </div>

            {/* ✅ Target Image 업로드를 위한 숨겨진 공용 Input */}
            <input
                type="file"
                hidden
                multiple
                ref={newTarInputRef}
                onChange={handleAddTar}
            />
        </div>
    )
}
/*
addPropertyControls(DetectDashboard, {
    apiEndpoint: {
        type: ControlType.String,
        title: "Backend API",
        defaultValue: "http://127.0.0.1:8000/api/ng-data",
    },
})
*/
