import {
    Select,
    SelectContent,
    SelectItem,
    SelectTrigger,
    SelectValue,
} from "@/components/ui/select"

interface ModelSelectorProps {
    value: string
    onChange: (value: string) => void
}

export function ModelSelector({ value, onChange }: ModelSelectorProps) {
    return (
        <Select value={value} onValueChange={onChange}>
            <SelectTrigger className="w-[180px]">
                <SelectValue placeholder="Select Model" />
            </SelectTrigger>
            <SelectContent>
                <SelectItem value="gpt-4">GPT-4</SelectItem>
                <SelectItem value="gpt-3.5-turbo">GPT-3.5 Turbo</SelectItem>
                <SelectItem value="claude-3-opus">Claude 3 Opus</SelectItem>
                <SelectItem value="claude-3-sonnet">Claude 3 Sonnet</SelectItem>
                <SelectItem value="ollama">Local Ollama</SelectItem>
            </SelectContent>
        </Select>
    )
} 