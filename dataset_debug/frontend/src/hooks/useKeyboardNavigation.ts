import { useEffect } from 'react';

interface KeyboardActions {
  onNextSample: () => void;
  onPrevSample: () => void;
  onNextCoil: () => void;
  onPrevCoil: () => void;
  onNextSlice: () => void;
  onPrevSlice: () => void;
  onNextSubject: () => void;
  onPrevSubject: () => void;
  onRandomSample: () => void;
  onJumpToIndex: () => void;
  onToggleAugment: () => void;
  onNewAugmentSeed: () => void;
  onToggleMasks: () => void;
  onTogglePenumbra: () => void;
  onToggleBackground: () => void;
  onToggleHistogram: () => void;
  onToggleDisplayMode: () => void;
  onToggleFullscreen: () => void;
  onToggleHelp: () => void;
  onExportAnnotations: () => void;
  onSaveAnnotations: () => void;
  onAddAnnotationField: () => void;
  onSelectDataset: (idx: number) => void;
}

export function useKeyboardNavigation(actions: KeyboardActions) {
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Ignore if inside an input or textarea
      if (['INPUT', 'TEXTAREA'].includes((e.target as HTMLElement).tagName)) {
        // Allow Esc to blur
        if (e.key === 'Escape') {
          (e.target as HTMLElement).blur();
        }
        return;
      }

      const hasCtrl = e.ctrlKey || e.metaKey;
      const hasShift = e.shiftKey;

      if (!hasCtrl && !hasShift && !e.altKey && /^[0-9]$/.test(e.key)) {
        const num = parseInt(e.key);
        if (num >= 1 && num <= 9) actions.onSelectDataset(num - 1);
        if (num === 0) actions.onSelectDataset(9);
        return;
      }

      if (!hasCtrl && !hasShift && !e.altKey && /^[qwertyuiopQWERTYUIOP]$/.test(e.key)) {
        const keys = ['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p'];
        const idx = keys.indexOf(e.key.toLowerCase());
        if (idx !== -1) actions.onSelectDataset(10 + idx);
        return;
      }

      switch (e.key) {
        case 'ArrowRight':
        case 'd':
        case 'D':
          e.preventDefault();
          actions.onNextSample();
          break;

        case 'ArrowLeft':
        case 'a':
        case 'A':
          e.preventDefault();
          actions.onPrevSample();
          break;

        case 'ArrowUp':
        case 'w':
        case 'W':
          e.preventDefault();
          actions.onNextSlice();
          break;

        case 'ArrowDown':
        case 's':
        case 'S':
          e.preventDefault();
          actions.onPrevSlice();
          break;

        case '.':
          actions.onNextSubject();
          break;

        case ',':
          actions.onPrevSubject();
          break;

        case ']':
          actions.onNextCoil();
          break;

        case '[':
          actions.onPrevCoil();
          break;
          
        case 'r':
        case 'R':
          actions.onRandomSample();
          break;
          
        case 'g':
        case 'G':
          e.preventDefault();
          actions.onJumpToIndex();
          break;
          
        case ' ':
          e.preventDefault();
          actions.onNewAugmentSeed();
          break;
          
        case 't':
        case 'T':
          actions.onToggleAugment();
          break;
          
        case 'm':
        case 'M':
          actions.onToggleMasks();
          break;
          
        case 'p':
        case 'P':
          actions.onTogglePenumbra();
          break;
          
        case 'b':
        case 'B':
          actions.onToggleBackground();
          break;
          
        case 'h':
        case 'H':
          actions.onToggleHistogram();
          break;
          
        case 'v':
        case 'V':
          actions.onToggleDisplayMode();
          break;
          
        case 'f':
        case 'F':
          actions.onToggleFullscreen();
          break;
          
        case 'k':
        case 'K':
        case '?':
          actions.onToggleHelp();
          break;
          
        case 'e':
        case 'E':
          actions.onExportAnnotations();
          break;
          
        case 'y':
        case 'Y':
          actions.onSaveAnnotations();
          break;
          
        case 'n':
        case 'N':
          actions.onAddAnnotationField();
          break;
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [actions]);
}
