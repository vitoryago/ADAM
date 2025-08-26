import { useIsMobile } from "@/hooks/use-mobile";
import { useEffect } from "react";

export function DebugDevice() {
  const isMobile = useIsMobile();
  
  useEffect(() => {
    console.log('=== Device Detection Debug ===');
    console.log('Window width:', window.innerWidth);
    console.log('Is Mobile (hook):', isMobile);
    console.log('User Agent:', navigator.userAgent);
    console.log('Touch support:', 'ontouchstart' in window);
    console.log('Hover support:', window.matchMedia('(hover: hover)').matches);
    console.log('Pointer:', window.matchMedia('(pointer: coarse)').matches ? 'coarse (touch)' : 'fine (mouse)');
    console.log('===========================');
  }, [isMobile]);
  
  return (
    <div className="fixed bottom-4 right-4 bg-black text-white p-4 rounded-lg z-50 text-xs">
      <div>Width: {window.innerWidth}px</div>
      <div>Mobile: {isMobile ? 'YES' : 'NO'}</div>
      <div>Touch: {'ontouchstart' in window ? 'YES' : 'NO'}</div>
      <div>Hover: {window.matchMedia('(hover: hover)').matches ? 'YES' : 'NO'}</div>
    </div>
  );
}