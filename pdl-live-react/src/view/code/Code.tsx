import { lazy, Suspense } from "react"

import { stringify } from "yaml"
import { match, P } from "ts-pattern"

import { type PdlBlock } from "../../pdl_ast"
import { map_block_children } from "../../pdl_ast_utils"

const Preview = lazy(() => import("./Preview"))
const PreviewLight = lazy(() => import("./PreviewLight"))
//import { type SupportedLanguage } from "./Preview"

export type SupportedLanguage =
  | "yaml"
  | "json"
  | "javascript"
  | "python"
  | "plaintext"

type Props = {
  block: PdlBlock
  language?: SupportedLanguage
  showLineNumbers?: boolean
  limitHeight?: boolean
  raw?: boolean
  remount?: boolean
  isWidthConstrained?: boolean
}

export default function Code({
  block,
  language = "yaml",
  showLineNumbers = false,
  limitHeight = true,
  raw = false,
  remount = false,
  isWidthConstrained = false,
}: Props) {
  const value =
    typeof block === "string"
      ? block
      : stringify(raw ? block : block_code_cleanup(block))

  return (
    <Suspense fallback={<div />}>
      {isWidthConstrained ? (
        <PreviewLight value={value} language={language || "yaml"} />
      ) : (
        <Preview
          limitHeight={limitHeight}
          showLineNumbers={showLineNumbers ?? false}
          language={language || "yaml"}
          remount={remount}
          isWidthConstrained={isWidthConstrained}
          value={value}
        />
      )}
    </Suspense>
  )
}

function block_code_cleanup(data: string | PdlBlock): string | PdlBlock {
  if (
    data === null ||
    typeof data === "boolean" ||
    typeof data === "number" ||
    typeof data === "string"
  ) {
    return data
  }
  // remove result
  const new_data = { ...data, result: undefined }
  // remove trace
  match(new_data).with({ trace: P._ }, (data) => {
    delete data.trace
  })
  // remove other trace artifacts
  delete new_data.id
  delete new_data.start_nanos
  delete new_data.end_nanos
  delete new_data.timezone
  delete new_data.context
  // remove contribute: ["result", context]
  if (
    new_data?.contribute?.includes("result") &&
    new_data?.contribute?.includes("context")
  ) {
    delete new_data.contribute
  }
  // remove empty defs list
  if (Object.keys(data?.defs ?? {}).length === 0) {
    delete new_data.defs
  }
  // remove location info
  delete new_data.location
  // recursive cleanup
  return map_block_children(block_code_cleanup, new_data)
}
