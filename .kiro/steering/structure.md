# Project Structure

## Root Directory
```
├── src/                 # Source code
├── public/              # Static assets (SVG icons)
├── .kiro/               # Kiro configuration and steering
├── node_modules/        # Dependencies
└── [config files]       # Various configuration files
```

## Source Organization
```
src/
└── app/                 # Next.js App Router directory
    ├── layout.tsx       # Root layout component
    ├── page.tsx         # Home page component
    ├── globals.css      # Global styles
    └── favicon.ico      # Site favicon
```

## Configuration Files
- `next.config.ts` - Next.js configuration (TypeScript)
- `tsconfig.json` - TypeScript compiler options
- `eslint.config.mjs` - ESLint configuration (ESM)
- `postcss.config.mjs` - PostCSS configuration (ESM)
- `package.json` - Dependencies and scripts

## Conventions
- Use TypeScript for all source files
- Configuration files use appropriate extensions (.ts, .mjs)
- App Router pattern (not Pages Router)
- Path aliases: `@/*` maps to `./src/*`
- Static assets in `public/` directory
- Global styles in `src/app/globals.css`

## File Naming
- React components: PascalCase (e.g., `layout.tsx`, `page.tsx`)
- Configuration files: kebab-case with appropriate extensions
- Static assets: lowercase with descriptive names