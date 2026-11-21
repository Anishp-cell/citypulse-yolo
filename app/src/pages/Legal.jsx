import { useTranslation } from 'react-i18next';

export function Privacy() {
  const { t } = useTranslation();
  return (
    <article className="stack">
      <h1>{t('legal.privacy_title')}</h1>
      {t('legal.privacy_body', { returnObjects: true }).map((p) => <p key={p}>{p}</p>)}
    </article>
  );
}

export function Terms() {
  const { t } = useTranslation();
  return (
    <article className="stack">
      <h1>{t('legal.terms_title')}</h1>
      <ol className="steps-list">{t('legal.terms_body', { returnObjects: true }).map((p) => <li key={p}>{p}</li>)}</ol>
    </article>
  );
}
