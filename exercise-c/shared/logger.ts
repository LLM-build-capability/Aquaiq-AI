export type LogLevel = "INFO" | "WARN" | "ERROR" | "DEBUG";

export interface LogEntry {
  timestamp: string;
  level: LogLevel;
  agent: string;
  correlation_id: string;
  causation_id?: string;
  capability?: string;
  message: string;
  [key: string]: unknown;
}

export function makeLogger(agentName: string) {
  function log(
    level: LogLevel,
    message: string,
    fields: Partial<Omit<LogEntry, "timestamp" | "level" | "agent" | "message">> & { [key: string]: unknown } = {}
  ) {
    const entry: LogEntry = {
      timestamp: new Date().toISOString(),
      level,
      agent: agentName,
      correlation_id: (fields.correlation_id as string | undefined) ?? "none",
      message,
      ...fields,
    };
    process.stdout.write(JSON.stringify(entry) + "\n");
  }

  return {
    info:  (msg: string, fields?: Parameters<typeof log>[2]) => log("INFO",  msg, fields),
    warn:  (msg: string, fields?: Parameters<typeof log>[2]) => log("WARN",  msg, fields),
    error: (msg: string, fields?: Parameters<typeof log>[2]) => log("ERROR", msg, fields),
    debug: (msg: string, fields?: Parameters<typeof log>[2]) => log("DEBUG", msg, fields),
  };
}

export type Logger = ReturnType<typeof makeLogger>;
