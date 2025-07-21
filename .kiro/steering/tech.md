# Technology Stack

## Core Framework
- **Next.js 15.4.2** - React framework with App Router
- **React 19.1.0** - UI library
- **TypeScript 5** - Type safety and development experience

## Styling & UI
- **Tailwind CSS 4** - Utility-first CSS framework
- **PostCSS** - CSS processing

## Content Processing
- **gray-matter** - YAML front-matter parser for markdown
- **remark** - Markdown processor
- **remark-html** - HTML output for remark
- **date-fns** - Date utility library

## Development Tools
- **ESLint** - Code linting with Next.js and TypeScript rules
- **TypeScript** - Static type checking

## Build Commands
```bash
# Development server
npm run dev

# Production build
npm run build

# Start production server
npm run start

# Lint code
npm run lint
```

## Configuration Notes
- Static export enabled for deployment
- Trailing slashes enforced
- Images unoptimized for static hosting
- Path aliases configured (`@/*` → `./src/*`)
- Strict TypeScript configuration