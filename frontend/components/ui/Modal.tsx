'use client';

import React, { useEffect, useId, useMemo, useRef } from 'react';

export type ModalProps = {
  open: boolean;
  onClose: () => void;
  title: string;
  children: React.ReactNode;
  closeOnBackdrop?: boolean;
  closeOnEsc?: boolean;
  className?: string;
};

const getFocusable = (root: HTMLElement): HTMLElement[] => {
  const selector = [
    'a[href]',
    'button:not([disabled])',
    'textarea:not([disabled])',
    'input:not([disabled])',
    'select:not([disabled])',
    '[tabindex]:not([tabindex="-1"])',
  ].join(',');

  return Array.from(root.querySelectorAll<HTMLElement>(selector)).filter((el) => {
    const style = window.getComputedStyle(el);
    return style.visibility !== 'hidden' && style.display !== 'none';
  });
};

export function Modal({
  open,
  onClose,
  title,
  children,
  closeOnBackdrop = true,
  closeOnEsc = true,
  className = '',
}: ModalProps) {
  const titleId = useId();
  const panelRef = useRef<HTMLDivElement | null>(null);
  const previouslyFocused = useRef<HTMLElement | null>(null);

  const labeledBy = useMemo(() => titleId, [titleId]);

  useEffect(() => {
    if (!open) return;

    previouslyFocused.current = document.activeElement as HTMLElement | null;

    // Defer focus until content is mounted.
    const t = window.setTimeout(() => {
      if (!panelRef.current) return;
      const focusables = getFocusable(panelRef.current);
      (focusables[0] || panelRef.current).focus();
    }, 0);

    return () => window.clearTimeout(t);
  }, [open]);

  useEffect(() => {
    if (!open) return;

    const onKeyDown = (e: KeyboardEvent) => {
      if (!panelRef.current) return;

      if (closeOnEsc && e.key === 'Escape') {
        e.preventDefault();
        onClose();
        return;
      }

      if (e.key !== 'Tab') return;

      const focusables = getFocusable(panelRef.current);
      if (focusables.length === 0) {
        e.preventDefault();
        panelRef.current.focus();
        return;
      }

      const first = focusables[0];
      const last = focusables[focusables.length - 1];
      const active = document.activeElement as HTMLElement | null;

      if (e.shiftKey) {
        if (!active || active === first || !panelRef.current.contains(active)) {
          e.preventDefault();
          last.focus();
        }
      } else {
        if (active === last) {
          e.preventDefault();
          first.focus();
        }
      }
    };

    document.addEventListener('keydown', onKeyDown);
    return () => document.removeEventListener('keydown', onKeyDown);
  }, [open, closeOnEsc, onClose]);

  useEffect(() => {
    if (open) return;
    if (!previouslyFocused.current) return;
    // Restore focus after close.
    previouslyFocused.current.focus?.();
    previouslyFocused.current = null;
  }, [open]);

  if (!open) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-end md:items-center justify-center">
      <div
        className="absolute inset-0 bg-black/60"
        onClick={closeOnBackdrop ? onClose : undefined}
      />
      <div
        ref={panelRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby={labeledBy}
        tabIndex={-1}
        className={`relative ui-surface-strong w-full md:w-3/4 lg:w-2/3 max-h-[80vh] overflow-auto p-6 z-10 outline-none ${className}`.trim()}
      >
        <div className="flex justify-between items-center mb-4">
          <h2 id={titleId} className="text-xl font-semibold">
            {title}
          </h2>
          <button type="button" onClick={onClose} className="text-slate-300 hover:underline underline-offset-4">
            Close
          </button>
        </div>
        {children}
      </div>
    </div>
  );
}
