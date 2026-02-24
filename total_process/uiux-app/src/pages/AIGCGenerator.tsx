import * as React from "react"
import { NavLink, useNavigate } from "react-router-dom"

// --- Types ---
type GenImage = { url: string; seed?: number; meta?: Record<string, any> }
type RefImage = { file: File; previewUrl: string }
type StoredGeneration = {
    version: number
    createdAt: number
    basePrompt: string
    enhancedPrompt: string
    settings: { ratio: string; size: number; batch: number }
    images: GenImage[]
    selectedIndex: number
}

const STORAGE_KEY = "rawer:last_generation_v1"
const FONT_STACK =
    '"Switzer","Noto Sans KR",Inter,system-ui,-apple-system,sans-serif'

// --- Utilities ---
function safeJsonParse<T>(s: string | null): T | null {
    if (!s) return null
    try {
        return JSON.parse(s) as T
    } catch {
        return null
    }
}

function normalizeEnhancedPrompt(raw: string): string {
    if (!raw) return ""
    const parts = raw.split("@").map((p) => p.trim()).filter(Boolean)
    const first = parts[0] || raw
    return first.replace(/^Prompt\s*\d+:\s*/i, "").trim()
}

async function postForm<T>(url: string, form: FormData): Promise<T> {
    const res = await fetch(url, { method: "POST", body: form })
    if (!res.ok) {
        const text = await res.text()
        throw new Error(text || `Request failed: ${res.status}`)
    }
    return (await res.json()) as T
}

// --- UI Atoms (포커스 유지를 위해 컴포넌트 외부로 이동) ---
const Card = ({
    title,
    children,
    style,
}: {
    title: string
    children: React.ReactNode
    style?: React.CSSProperties
}) => (
    <div
        style={{
            border: "1px solid rgba(0,0,0,0.10)",
            borderRadius: 16,
            padding: 16,
            background: "white",
            boxShadow: "0 1px 2px rgba(0,0,0,0.04)",
            ...style,
        }}
    >
        <div style={{ fontWeight: 500, fontSize: 14, marginBottom: 10 }}>
            {title}
        </div>
        {children}
    </div>
)

const Btn = ({ label, onClick, disabled, strong, style }: any) => (
    <button
        onClick={onClick}
        disabled={disabled}
        style={{
            border: "1px solid rgba(0,0,0,0.12)",
            background: strong ? "rgba(0,0,0,0.90)" : "white",
            color: strong ? "white" : "black",
            borderRadius: 12,
            padding: "10px 12px",
            fontWeight: 500,
            cursor: disabled ? "not-allowed" : "pointer",
            opacity: disabled ? 0.5 : 1,
            fontSize: 14,
            fontFamily: FONT_STACK,
            ...style,
        }}
    >
        {label}
    </button>
)

type AigcGeneratorProps = {
    apiBase?: string
    detectRoute?: string
    defaultBatch?: number
    defaultRatio?: string
    defaultSize?: number
}

export default function CombinedRunBuilder(props: AigcGeneratorProps) {
    const navigate = useNavigate()
    const apiBase = (props.apiBase || window.location.origin).replace(/\/$/, "")
    const linkStyle = ({ isActive }: { isActive: boolean }) => ({
        padding: "6px 10px",
        borderRadius: 8,
        fontSize: 11,
        fontWeight: 700,
        textDecoration: "none",
        color: isActive ? "#000" : "#666",
        background: isActive ? "#F2F2F2" : "transparent",
    })

    // --- State ---
    const [baseText, setBaseText] = React.useState("")
    const [enhancedText, setEnhancedText] = React.useState("")
    const [ratio, setRatio] = React.useState(props.defaultRatio || "2:3")
    const [size] = React.useState(props.defaultSize || 4096)
    const [batch] = React.useState(props.defaultBatch || 6)
    const [images, setImages] = React.useState<GenImage[]>([])
    const [selectedIndex, setSelectedIndex] = React.useState(0)
    const [loadingEnhance, setLoadingEnhance] = React.useState(false)
    const [loadingGenerate, setLoadingGenerate] = React.useState(false)
    const [error, setError] = React.useState("")
    const [refImages, setRefImages] = React.useState<RefImage[]>([])

    const refInputRef = React.useRef<HTMLInputElement>(null)

    // 1. 초기 로드
    React.useEffect(() => {
        const stored = safeJsonParse<StoredGeneration>(
            localStorage.getItem(STORAGE_KEY)
        )
        if (!stored) return
        setBaseText(stored.basePrompt || "")
        setEnhancedText(stored.enhancedPrompt || "")
        setRatio(stored.settings?.ratio ?? props.defaultRatio ?? "2:3")
        setImages(stored.images || [])
        setSelectedIndex(stored.selectedIndex || 0)
    }, [])

    // 2. 자동 저장 (데이터 유실 방지 및 다음 단계 전달용)
    // CombinedRunBuilder 내부의 자동 저장 useEffect 수정
    React.useEffect(() => {
        const payload = {
            version: 1,
            createdAt: Date.now(),
            basePrompt: baseText,
            enhancedPrompt: enhancedText,
            settings: { ratio, size, batch },
            images: images, // 생성된 이미지들 (Target용)
            selectedIndex: selectedIndex,
            // ✅ 추가: 사용자가 업로드한 참조 이미지의 URL들을 저장합니다.
            referenceImageUrls: refImages.map((r) => r.previewUrl),
        }
        localStorage.setItem(STORAGE_KEY, JSON.stringify(payload))
    }, [
        baseText,
        enhancedText,
        ratio,
        size,
        batch,
        images,
        selectedIndex,
        refImages,
    ]) // refImages를 의존성 배열에 추가

    const onEnhance = async () => {
        if (!baseText.trim()) return setError("Base prompt 미입력 상태")
        setError("")
        setLoadingEnhance(true)
        try {
            const form = new FormData()
            form.append("prompt", baseText)
            form.append(
                "options",
                JSON.stringify({ lang: "ko", length: "medium" })
            )
            refImages.forEach((r) => form.append("ref_images", r.file))
            const data = await postForm<{ enhanced_prompt: string }>(
                `${apiBase}/api/enhance`,
                form
            )
            setEnhancedText(normalizeEnhancedPrompt(data.enhanced_prompt || ""))
        } catch (e: any) {
            setError(e?.message || "Enhance 실패")
        } finally {
            setLoadingEnhance(false)
        }
    }

    const onGenerate = async () => {
        const promptToUse = enhancedText || baseText
        if (!promptToUse.trim()) return setError("Prompt 미입력 상태")
        setError("")
        setLoadingGenerate(true)
        try {
            const form = new FormData()
            form.append("prompt", promptToUse)
            form.append("ratio", ratio)
            form.append("size", String(size))
            form.append("n", String(batch))
            refImages.forEach((r) => form.append("ref_images", r.file))
            const data = await postForm<{ images: GenImage[] }>(
                `${apiBase}/api/generate`,
                form
            )
            setImages(data.images || [])
            setSelectedIndex(0)
        } catch (e: any) {
            setError(e?.message || "Generate 실패")
        } finally {
            setLoadingGenerate(false)
        }
    }

    return (
        <div
            style={{
                width: "100%",
                height: "100vh",
                minHeight: "100vh",
                fontFamily: FONT_STACK,
                display: "flex",
                flexDirection: "column",
                background: "#FFFFFF",
                overflow: "hidden",
            }}
        >
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
                    AIGC Generator
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
                    display: "flex",
                    gap: 20,
                    padding: 24,
                    flex: 1,
                    minHeight: 0,
                    alignItems: "stretch",
                    boxSizing: "border-box",
                    width: "100%",
                }}
            >
                {/* Left Section */}
                <div
                    style={{
                        flex: 2,
                        display: "flex",
                        flexDirection: "column",
                        gap: 16,
                        overflowY: "auto",
                        paddingRight: 8,
                        minHeight: 0,
                    }}
                >
                    <div style={{ fontSize: 20, fontWeight: 600 }}>
                        Prompt Enhancer
                    </div>
                    {error && (
                        <div
                            style={{
                                padding: 12,
                                borderRadius: 12,
                                background: "rgba(220,38,38,0.06)",
                                color: "red",
                                fontSize: 12,
                            }}
                        >
                            {error}
                        </div>
                    )}

                    <Card title="Reference Images">
                        <input
                            ref={refInputRef}
                            type="file"
                            accept="image/*"
                            multiple
                            hidden
                            onChange={(e) => {
                                const files = Array.from(
                                    e.target.files || []
                                ).map((f) => ({
                                    file: f,
                                    previewUrl: URL.createObjectURL(f),
                                }))
                                setRefImages((prev) =>
                                    [...prev, ...files].slice(0, 6)
                                )
                            }}
                        />
                        <div style={{ display: "flex", gap: 10 }}>
                            <Btn
                                label="파일 선택"
                                onClick={() => refInputRef.current?.click()}
                            />
                            <Btn
                                label="Clear"
                                onClick={() => setRefImages([])}
                                disabled={!refImages.length}
                            />
                        </div>
                        <div
                            style={{
                                display: "grid",
                                gridTemplateColumns: "repeat(6, 1fr)",
                                gap: 8,
                                marginTop: 12,
                            }}
                        >
                            {refImages.map((r, i) => (
                                <img
                                    key={i}
                                    src={r.previewUrl}
                                    style={{
                                        width: "100%",
                                        aspectRatio: "1/1",
                                        objectFit: "cover",
                                        borderRadius: 12,
                                        border: "1px solid #eee",
                                    }}
                                />
                            ))}
                        </div>
                    </Card>

                    <Card title="Base Prompt">
                        <textarea
                            placeholder="이미지 설명을 입력하세요..."
                            value={baseText}
                            onChange={(e) => setBaseText(e.target.value)}
                            style={{
                                width: "100%",
                                maxWidth: "100%",
                                minHeight: 100,
                                borderRadius: 14,
                                padding: 14,
                                border: "1px solid #ddd",
                                outline: "none",
                                resize: "none",
                                boxSizing: "border-box",
                            }}
                        />
                        <Btn
                            label={
                                loadingEnhance
                                    ? "Enhancing..."
                                    : "Enhance Prompt"
                            }
                            onClick={onEnhance}
                            disabled={loadingEnhance}
                            strong
                            style={{ marginTop: 12 }}
                        />
                    </Card>

                    <Card title="Enhanced Prompt (Editable)">
                        <textarea
                            placeholder="Enhance 결과 표시 구역"
                            value={enhancedText}
                            onChange={(e) => setEnhancedText(e.target.value)}
                            style={{
                                width: "100%",
                                maxWidth: "100%",
                                minHeight: 140,
                                borderRadius: 14,
                                padding: 14,
                                border: "1px solid #ddd",
                                outline: "none",
                                resize: "none",
                                boxSizing: "border-box",
                            }}
                        />
                    </Card>
                </div>

                {/* Right Section */}
                <div
                    style={{
                        flex: 1,
                        display: "flex",
                        flexDirection: "column",
                        gap: 16,
                        minWidth: 320,
                        minHeight: 0,
                        overflowY: "auto",
                    }}
                >
                    <div style={{ fontSize: 20, fontWeight: 600 }}>
                        IMG Generator
                    </div>
                    <Card title="Settings">
                        <div
                            style={{
                                display: "flex",
                                flexDirection: "column",
                                gap: 10,
                            }}
                        >
                            <div
                                style={{
                                    display: "flex",
                                    justifyContent: "space-between",
                                    alignItems: "center",
                                    fontSize: 13,
                                }}
                            >
                                <span>Ratio</span>
                                <select
                                    value={ratio}
                                    onChange={(e) => setRatio(e.target.value)}
                                    style={{ padding: 4 }}
                                >
                                    {["1:1", "4:5", "2:3", "16:9"].map((r) => (
                                        <option key={r} value={r}>
                                            {r}
                                        </option>
                                    ))}
                                </select>
                            </div>
                            <Btn
                                label={
                                    loadingGenerate
                                        ? "Generating..."
                                        : "Generate Images"
                                }
                                onClick={onGenerate}
                                disabled={loadingGenerate}
                                strong
                            />
                        </div>
                    </Card>

                    <Card
                        title="Results"
                        style={{
                            flex: 1,
                            position: "relative",
                            paddingBottom: 70,
                            minHeight: 250,
                        }}
                    >
                        <div style={{ height: "100%", overflowY: "auto" }}>
                            {images.length === 0 ? (
                                <div
                                    style={{
                                        fontSize: 13,
                                        opacity: 0.5,
                                        textAlign: "center",
                                        padding: "40px 0",
                                    }}
                                >
                                    결과 없음
                                </div>
                            ) : (
                                <div
                                    style={{
                                        display: "grid",
                                        gridTemplateColumns: "1fr 1fr",
                                        gap: 8,
                                    }}
                                >
                                    {images.map((img, idx) => (
                                        <button
                                            key={idx}
                                            onClick={() =>
                                                setSelectedIndex(idx)
                                            }
                                            style={{
                                                padding: 0,
                                                border:
                                                    selectedIndex === idx
                                                        ? "2px solid black"
                                                        : "1px solid #eee",
                                                borderRadius: 8,
                                                overflow: "hidden",
                                                cursor: "pointer",
                                                background: "none",
                                            }}
                                        >
                                            <img
                                                src={img.url}
                                                style={{
                                                    width: "100%",
                                                    display: "block",
                                                }}
                                            />
                                        </button>
                                    ))}
                                </div>
                            )}
                        </div>
                        <div
                            style={{
                                position: "absolute",
                                bottom: 16,
                                left: 16,
                                right: 16,
                            }}
                        >
                            <Btn
                                label="Open Detect"
                                onClick={() =>
                                    navigate(props.detectRoute || "/detect")
                                }
                                disabled={images.length === 0}
                                strong
                                style={{ width: "100%" }}
                            />
                        </div>
                    </Card>
                </div>
            </div>
        </div>
    )
}
