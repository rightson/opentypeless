import { useEffect, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { Plus, X } from 'lucide-react'
import type {
  LlmGatewayAuthMode,
  LlmGatewayConfig,
  LlmGatewayHeader,
  LlmGatewayProtocol,
} from '../../stores/appStore'
import { FormField } from './shared/FormField'

const inputClass =
  'w-full px-3 py-2.5 bg-bg-secondary border border-border rounded-[10px] text-[13px] text-text-primary outline-none focus:border-border-focus transition-colors'

const PROTOCOL_DEFAULT_AUTH: Record<LlmGatewayProtocol, { header: string; prefix: string }> = {
  openai: { header: 'Authorization', prefix: 'Bearer ' },
  anthropic: { header: 'x-api-key', prefix: '' },
  gemini: { header: 'x-goog-api-key', prefix: '' },
}

interface ProtocolSectionProps {
  gateway: LlmGatewayConfig
  onChange: (patch: Partial<LlmGatewayConfig>) => void
}

/** Protocol + authentication source selectors (rendered above the API key field). */
export function GatewayProtocolSection({ gateway, onChange }: ProtocolSectionProps) {
  const { t } = useTranslation()
  return (
    <>
      <p className="text-[12px] leading-relaxed text-text-tertiary">
        {t('settings.gateway.description')}
      </p>
      <FormField label={t('settings.gateway.protocol')}>
        <select
          aria-label={t('settings.gateway.protocol')}
          value={gateway.protocol}
          onChange={(e) => onChange({ protocol: e.target.value as LlmGatewayProtocol })}
          className={inputClass}
        >
          <option value="openai">{t('settings.gateway.protocolOpenai')}</option>
          <option value="anthropic">{t('settings.gateway.protocolAnthropic')}</option>
          <option value="gemini">{t('settings.gateway.protocolGemini')}</option>
        </select>
      </FormField>
      <FormField label={t('settings.gateway.authMode')}>
        <select
          aria-label={t('settings.gateway.authMode')}
          value={gateway.auth_mode}
          onChange={(e) => onChange({ auth_mode: e.target.value as LlmGatewayAuthMode })}
          className={inputClass}
        >
          <option value="static">{t('settings.gateway.authStatic')}</option>
          <option value="command">{t('settings.gateway.authCommand')}</option>
          <option value="none">{t('settings.gateway.authNone')}</option>
        </select>
      </FormField>
    </>
  )
}

function parseArgs(text: string): string[] {
  return text.split(/\r?\n/).filter((line) => line.length > 0)
}

/** Key helper, auth header and extra header settings (rendered below the base URL). */
export function GatewayAdvancedSection({ gateway, onChange }: ProtocolSectionProps) {
  const { t } = useTranslation()
  const defaults = PROTOCOL_DEFAULT_AUTH[gateway.protocol] ?? PROTOCOL_DEFAULT_AUTH.openai
  // The textarea keeps its own text so blank lines can be typed; the config
  // stores one argument per non-empty line.
  const [argsText, setArgsText] = useState(gateway.key_command_args.join('\n'))
  useEffect(() => {
    setArgsText((current) =>
      parseArgs(current).join('\n') === gateway.key_command_args.join('\n')
        ? current
        : gateway.key_command_args.join('\n'),
    )
  }, [gateway.key_command_args])

  const updateHeader = (index: number, patch: Partial<LlmGatewayHeader>) => {
    onChange({
      extra_headers: gateway.extra_headers.map((header, i) =>
        i === index ? { ...header, ...patch } : header,
      ),
    })
  }

  return (
    <>
      {gateway.auth_mode === 'command' && (
        <div className="space-y-3 rounded-[10px] border border-border px-3 py-3">
          <FormField label={t('settings.gateway.keyCommand')}>
            <input
              aria-label={t('settings.gateway.keyCommand')}
              value={gateway.key_command}
              onChange={(e) => onChange({ key_command: e.target.value })}
              placeholder={t('settings.gateway.keyCommandPlaceholder')}
              spellCheck={false}
              className={`${inputClass} font-mono`}
            />
            <p className="mt-1 text-[11px] leading-relaxed text-text-tertiary">
              {t('settings.gateway.keyCommandHint')}
            </p>
          </FormField>
          <FormField label={t('settings.gateway.keyCommandArgs')}>
            <textarea
              aria-label={t('settings.gateway.keyCommandArgs')}
              value={argsText}
              onChange={(e) => {
                setArgsText(e.target.value)
                onChange({ key_command_args: parseArgs(e.target.value) })
              }}
              rows={3}
              spellCheck={false}
              placeholder={t('settings.gateway.keyCommandArgsPlaceholder')}
              className={`${inputClass} resize-y font-mono`}
            />
            <p className="mt-1 text-[11px] leading-relaxed text-text-tertiary">
              {t('settings.gateway.keyCommandArgsHint')}
            </p>
          </FormField>
          <FormField label={t('settings.gateway.keyJsonField')}>
            <input
              aria-label={t('settings.gateway.keyJsonField')}
              value={gateway.key_command_json_field}
              onChange={(e) => onChange({ key_command_json_field: e.target.value })}
              placeholder={t('settings.gateway.keyJsonFieldPlaceholder')}
              spellCheck={false}
              className={`${inputClass} font-mono`}
            />
          </FormField>
          <div className="grid grid-cols-2 gap-3">
            <FormField label={t('settings.gateway.keyCacheSeconds')}>
              <input
                aria-label={t('settings.gateway.keyCacheSeconds')}
                type="number"
                min={0}
                max={86400}
                value={gateway.key_cache_seconds}
                onChange={(e) =>
                  onChange({ key_cache_seconds: Math.max(0, Number(e.target.value) || 0) })
                }
                className={inputClass}
              />
            </FormField>
            <FormField label={t('settings.gateway.keyTimeoutSeconds')}>
              <input
                aria-label={t('settings.gateway.keyTimeoutSeconds')}
                type="number"
                min={1}
                max={120}
                value={gateway.key_command_timeout_seconds}
                onChange={(e) =>
                  onChange({
                    key_command_timeout_seconds: Math.max(1, Number(e.target.value) || 1),
                  })
                }
                className={inputClass}
              />
            </FormField>
          </div>
        </div>
      )}

      {gateway.auth_mode !== 'none' && (
        <div className="grid grid-cols-[2fr_1fr] gap-3">
          <FormField label={t('settings.gateway.authHeader')}>
            <input
              aria-label={t('settings.gateway.authHeader')}
              value={gateway.auth_header}
              onChange={(e) => onChange({ auth_header: e.target.value })}
              placeholder={defaults.header}
              spellCheck={false}
              className={`${inputClass} font-mono`}
            />
          </FormField>
          <FormField label={t('settings.gateway.authPrefix')}>
            <input
              aria-label={t('settings.gateway.authPrefix')}
              value={gateway.auth_prefix}
              onChange={(e) => onChange({ auth_prefix: e.target.value })}
              placeholder={gateway.auth_header.trim() ? '' : defaults.prefix.trim()}
              disabled={!gateway.auth_header.trim()}
              spellCheck={false}
              className={`${inputClass} font-mono disabled:opacity-50`}
            />
          </FormField>
        </div>
      )}

      <FormField label={t('settings.gateway.extraHeaders')}>
        <div className="space-y-2">
          {gateway.extra_headers.map((header, index) => (
            <div key={index} className="flex gap-2">
              <input
                aria-label={t('settings.gateway.headerName')}
                value={header.name}
                onChange={(e) => updateHeader(index, { name: e.target.value })}
                placeholder={t('settings.gateway.headerName')}
                spellCheck={false}
                className={`${inputClass} min-w-0 flex-[2] font-mono`}
              />
              <input
                aria-label={t('settings.gateway.headerValue')}
                value={header.value}
                onChange={(e) => updateHeader(index, { value: e.target.value })}
                placeholder={t('settings.gateway.headerValue')}
                spellCheck={false}
                className={`${inputClass} min-w-0 flex-[3] font-mono`}
              />
              <button
                type="button"
                aria-label={t('settings.gateway.removeHeader')}
                title={t('settings.gateway.removeHeader')}
                onClick={() =>
                  onChange({ extra_headers: gateway.extra_headers.filter((_, i) => i !== index) })
                }
                className="grid w-9 flex-none place-items-center rounded-[10px] border border-border bg-bg-secondary text-text-tertiary hover:border-border-focus hover:text-text-primary"
              >
                <X size={14} />
              </button>
            </div>
          ))}
          <button
            type="button"
            onClick={() =>
              onChange({ extra_headers: [...gateway.extra_headers, { name: '', value: '' }] })
            }
            className="flex items-center gap-1.5 rounded-[10px] border border-dashed border-border px-3 py-2 text-[12px] text-text-secondary hover:border-border-focus hover:text-text-primary"
          >
            <Plus size={13} />
            {t('settings.gateway.addHeader')}
          </button>
          <p className="text-[11px] leading-relaxed text-text-tertiary">
            {t('settings.gateway.extraHeadersHint')}
          </p>
        </div>
      </FormField>
    </>
  )
}
