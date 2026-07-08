"use client";

import { useEffect, useRef } from "react";

interface SelectAllCheckboxProps {
  checked: boolean;
  indeterminate: boolean;
  onChange: () => void;
  disabled?: boolean;
}

/**
 * SelectAllCheckbox
 *
 * React cannot set `indeterminate` on a checkbox via JSX props — it must be
 * applied imperatively via a DOM ref. This wrapper handles that correctly.
 */
export function SelectAllCheckbox({
  checked,
  indeterminate,
  onChange,
  disabled = false,
}: SelectAllCheckboxProps) {
  const ref = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (ref.current) {
      ref.current.indeterminate = indeterminate;
    }
  }, [indeterminate]);

  return (
    <input
      ref={ref}
      type="checkbox"
      checked={checked}
      onChange={onChange}
      disabled={disabled}
      style={{ accentColor: "#7c3aed", width: 15, height: 15, cursor: disabled ? "default" : "pointer" }}
    />
  );
}
