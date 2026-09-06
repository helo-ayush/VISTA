/**
 * Resolves static asset URLs relative to the current application base path.
 * Ensures models, dicts, and sample images resolve correctly whether deployed at
 * root (https://domain.com/), on a subpath (https://domain.com/VISTA/), or local dev.
 *
 * @param {string} relPath - Relative path to the asset
 * @returns {string} - Fully resolved path
 */
export function getAssetUrl(relPath) {
  const base = import.meta.env.BASE_URL || './';
  const cleanBase = base.endsWith('/') ? base : `${base}/`;
  const cleanRel = relPath.startsWith('/') ? relPath.slice(1) : relPath;
  return `${cleanBase}${cleanRel}`;
}
