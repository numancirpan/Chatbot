export default function LoadingState() {
  return (
    <div className="loading-state" role="status" aria-live="polite">
      <div className="loading-copy">
        <span>Asistan yanıt hazırlıyor</span>
        <small>Resmi kaynaklar kontrol ediliyor</small>
      </div>
      <div className="loading-dots" aria-hidden="true">
        <span />
        <span />
        <span />
      </div>
    </div>
  );
}
