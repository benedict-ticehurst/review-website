// Prefix a root-relative path with the site's base path (e.g. /review-website on GitHub Pages).
export function withBase(path: string): string {
	return `${import.meta.env.BASE_URL.replace(/\/$/, '')}${path}`;
}
