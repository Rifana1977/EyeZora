import { useState, useCallback } from "react";

/**
 * useRowSelection — generic hook for table row bulk-selection.
 *
 * Usage:
 *   const sel = useRowSelection();
 *   sel.toggleOne(id)
 *   sel.toggleAll(allVisibleIds)
 *   sel.clearSelection()
 *   sel.selected          // Set<string>
 *   sel.selectedCount     // number
 *   sel.selectedArray     // string[]
 *   sel.isAllSelected(ids)
 *   sel.isIndeterminate(ids)
 */
export function useRowSelection() {
  const [selected, setSelected] = useState<Set<string>>(new Set());

  const toggleOne = useCallback((id: string) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
  }, []);

  /**
   * Toggle all rows in the provided list.
   * If every ID in `ids` is already selected → deselect all.
   * Otherwise → select all.
   */
  const toggleAll = useCallback((ids: string[]) => {
    setSelected((prev) => {
      const allSelected = ids.length > 0 && ids.every((id) => prev.has(id));
      if (allSelected) {
        // Deselect all from this page
        const next = new Set(prev);
        ids.forEach((id) => next.delete(id));
        return next;
      } else {
        // Select all from this page (additive — preserves other pages)
        const next = new Set(prev);
        ids.forEach((id) => next.add(id));
        return next;
      }
    });
  }, []);

  const clearSelection = useCallback(() => {
    setSelected(new Set());
  }, []);

  const isAllSelected = useCallback(
    (ids: string[]) => ids.length > 0 && ids.every((id) => selected.has(id)),
    [selected]
  );

  const isIndeterminate = useCallback(
    (ids: string[]) => ids.some((id) => selected.has(id)) && !ids.every((id) => selected.has(id)),
    [selected]
  );

  return {
    selected,
    selectedCount: selected.size,
    selectedArray: Array.from(selected),
    toggleOne,
    toggleAll,
    clearSelection,
    isAllSelected,
    isIndeterminate,
  };
}
