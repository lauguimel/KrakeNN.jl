import { defineConfig } from 'vitepress'
import { tabsMarkdownPlugin } from 'vitepress-plugin-tabs'
import mathjax3 from "markdown-it-mathjax3";
import footnote from "markdown-it-footnote";
import path from 'path'
import krkGrammar from './krk.tmLanguage.json'

function getBaseRepository(base: string): string {
  if (!base || base === '/') return '/';
  const parts = base.split('/').filter(Boolean);
  return parts.length > 0 ? `/${parts[0]}/` : '/';
}

const baseTemp = {
  base: 'REPLACE_ME_DOCUMENTER_VITEPRESS',
}

const navTemp = {
  nav: 'REPLACE_ME_DOCUMENTER_VITEPRESS',
}

// DocumenterVitepress fills both `nav` and `sidebar` from `makedocs(pages=...)`.
// Keep the generated top-nav as-is, but show only relevant top-level groups in
// the left sidebar for each URL prefix.
const sidebarHolder = {
  sidebar: 'REPLACE_ME_DOCUMENTER_VITEPRESS',
}
const flatSidebar: any[] = sidebarHolder.sidebar as unknown as any[]

function section(name: string): any[] {
  const found = flatSidebar.find(x => x && x.text === name)
  return found ? [found] : []
}

function sections(...names: string[]): any[] {
  return names.flatMap(name => section(name))
}

const sectionedSidebar: Record<string, any[]> = {
  // Guide
  '/installation': section('Guide'),
  '/getting_started': section('Guide'),
  '/concepts_index': section('Guide'),
  '/capabilities': section('Guide'),
  '/architecture': section('Guide'),
  '/users/krk-reference': section('Guide'),
  '/users/incompressible-navier-stokes': section('Guide'),
  '/users/simulation-state-checkpoints': section('Guide'),
  // Examples (grouped by physics; two pages re-homed from users/tutorials/)
  '/users/tutorials/': section('Examples'),
  '/examples/': section('Examples'),
  // Benchmarks (validation cases + performance)
  '/users/benchmarks/': section('Benchmarks'),
  '/benchmarks/': section('Benchmarks'),
  // Reference (.krk DSL + API + Julia API + Theory)
  '/krk/': section('Reference'),
  '/api/': section('Reference'),
  '/theory/': section('Reference'),
  '/references': section('Reference'),
  '/': flatSidebar,
}

const nav = [
  ...navTemp.nav as unknown as any[],
  {
    component: 'VersionPicker'
  }
]

export default defineConfig({
  base: 'REPLACE_ME_DOCUMENTER_VITEPRESS',
  title: 'REPLACE_ME_DOCUMENTER_VITEPRESS',
  description: 'REPLACE_ME_DOCUMENTER_VITEPRESS',
  lastUpdated: true,
  cleanUrls: true,
  outDir: 'REPLACE_ME_DOCUMENTER_VITEPRESS',
  ignoreDeadLinks: [
    // Downloadable .krk example files are served as raw assets, not VitePress
    // routes, so the dead-link checker cannot resolve them (works under Documenter).
    /\.krk$/,
    // Pre-existing unresolved Documenter @ref cross-reference (krk-reference Presets).
    /@ref$/,
  ],
  head: [
    ['link', { rel: 'icon', href: 'REPLACE_ME_DOCUMENTER_VITEPRESS_FAVICON' }],
    ['script', {src: `${getBaseRepository(baseTemp.base)}versions.js`}],
    ['script', {src: `${baseTemp.base}siteinfo.js`}]
  ],

  vite: {
    define: {
      __DEPLOY_ABSPATH__: JSON.stringify('REPLACE_ME_DOCUMENTER_VITEPRESS_DEPLOY_ABSPATH'),
    },
    resolve: {
      alias: {
        '@': path.resolve(__dirname, '../components')
      }
    },
    optimizeDeps: {
      exclude: [
        '@nolebase/vitepress-plugin-enhanced-readabilities/client',
        'vitepress',
        '@nolebase/ui',
      ],
    },
    ssr: {
      noExternal: [
        '@nolebase/vitepress-plugin-enhanced-readabilities',
        '@nolebase/ui',
      ],
    },
  },
  markdown: {
    math: true,
    config(md) {
      md.use(tabsMarkdownPlugin),
      md.use(mathjax3),
      md.use(footnote)
    },
    // Register the .krk TextMate grammar so ```krk fenced blocks are highlighted.
    // Kept inside the `markdown` object so DocumenterVitepress's REPLACE_ME
    // substitution preserves it.
    languages: [
      { ...(krkGrammar as any), name: 'krk', scopeName: 'source.krk' }
    ],
    theme: {
      light: "github-light",
      dark: "github-dark"}
  },
  themeConfig: {
    outline: 'deep',
    logo: 'REPLACE_ME_DOCUMENTER_VITEPRESS',
    search: {
      provider: 'local',
      options: {
        detailedView: true
      }
    },
    nav,
    sidebar: sectionedSidebar,
    editLink: 'REPLACE_ME_DOCUMENTER_VITEPRESS',
    socialLinks: [
      { icon: 'github', link: 'REPLACE_ME_DOCUMENTER_VITEPRESS' }
    ],
    footer: {
      message: 'Made with <a href="https://luxdl.github.io/DocumenterVitepress.jl/dev/" target="_blank"><strong>DocumenterVitepress.jl</strong></a><br>',
      copyright: `© Copyright ${new Date().getUTCFullYear()}.`
    }
  }
})
