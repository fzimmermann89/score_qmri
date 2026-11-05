import { useEffect, useState } from 'react'

export type ToastType = 'success' | 'error' | 'info'

interface ToastMessage {
  id: number
  message: string
  type: ToastType
}

let toastId = 0

// Global toast state management
const listeners: ((toast: ToastMessage | null) => void)[] = []
let currentToast: ToastMessage | null = null

export function showToast(message: string, type: ToastType = 'info') {
  currentToast = { id: ++toastId, message, type }
  listeners.forEach((listener) => listener(currentToast))
}

function hideToast() {
  currentToast = null
  listeners.forEach((listener) => listener(null))
}

export function Toast() {
  const [toast, setToast] = useState<ToastMessage | null>(null)

  useEffect(() => {
    listeners.push(setToast)
    return () => {
      const idx = listeners.indexOf(setToast)
      if (idx >= 0) listeners.splice(idx, 1)
    }
  }, [])

  useEffect(() => {
    if (toast) {
      const timer = setTimeout(hideToast, 3000)
      return () => clearTimeout(timer)
    }
  }, [toast])

  if (!toast) return null

  const bgColor = {
    success: 'bg-green-600',
    error: 'bg-red-600',
    info: 'bg-blue-600',
  }[toast.type]

  return (
    <div className="fixed bottom-4 right-4 z-50 animate-slide-up">
      <div
        className={`${bgColor} text-white px-4 py-2 rounded-lg shadow-lg flex items-center gap-2`}
      >
        {toast.type === 'success' && <span>✓</span>}
        {toast.type === 'error' && <span>✕</span>}
        {toast.type === 'info' && <span>ℹ</span>}
        <span>{toast.message}</span>
        <button
          onClick={hideToast}
          className="ml-2 opacity-60 hover:opacity-100"
        >
          ×
        </button>
      </div>
    </div>
  )
}
