# Design Document

## Overview

This design document outlines the architecture for a personal blog platform built with Next.js 15.4.2, leveraging static site generation for optimal performance and deployment flexibility. The blog will process markdown content with YAML front matter, provide a responsive reading experience, and include search/filtering capabilities.

The design prioritizes simplicity, performance, and maintainability while utilizing the existing technology stack including gray-matter, remark, remark-html, and date-fns.

## Architecture

### Static Site Generation Approach
- **Static Export**: Leveraging Next.js static export capability for deployment to GitHub Pages
- **Build-time Processing**: All markdown files processed during build time for optimal performance
- **File-based Routing**: Using Next.js App Router for clean URL structure

### Content Processing Pipeline
```
Markdown Files → gray-matter → Front Matter + Content → remark → remark-html → Static HTML
```

### Directory Structure
```
src/
├── app/
│   ├── layout.tsx           # Root layout with header/footer
│   ├── page.tsx             # Homepage with post list
│   ├── posts/
│   │   └── [slug]/
│   │       └── page.tsx     # Individual post pages
│   ├── about/
│   │   └── page.tsx         # About page
│   └── globals.css          # Global styles
├── components/
│   ├── PostCard.tsx         # Post preview component
│   ├── PostContent.tsx      # Full post display component
│   ├── SearchBar.tsx        # Search functionality
│   ├── TagFilter.tsx        # Tag filtering component
│   └── Header.tsx           # Site header with navigation
├── lib/
│   ├── posts.ts             # Post processing utilities
│   ├── search.ts            # Search functionality
│   └── types.ts             # TypeScript type definitions
└── content/
    └── posts/               # Markdown blog posts
        ├── post-1.md
        └── post-2.md
```

## Components and Interfaces

### Core Data Types
```typescript
interface BlogPost {
  slug: string;
  title: string;
  date: string;
  author: string;
  excerpt: string;
  tags: string[];
  content: string;
  readTime: number;
}

interface PostMetadata {
  title: string;
  date: string;
  author: string;
  excerpt?: string;
  tags?: string[];
}
```

### Key Components

#### PostCard Component
- **Purpose**: Display post previews on homepage
- **Props**: BlogPost metadata
- **Features**: Title, date, excerpt, read time, tags
- **Responsive**: Card layout adapting to screen size

#### PostContent Component
- **Purpose**: Render full blog post content
- **Props**: BlogPost with full content
- **Features**: Markdown rendering, syntax highlighting, navigation
- **SEO**: Dynamic meta tags and Open Graph data

#### SearchBar Component
- **Purpose**: Client-side search functionality
- **State**: Search query, filtered results
- **Features**: Real-time filtering, highlight matching terms
- **Performance**: Debounced search to avoid excessive filtering

#### TagFilter Component
- **Purpose**: Filter posts by category tags
- **State**: Selected tags, available tags
- **Features**: Multi-select filtering, tag counts
- **Integration**: Works with search functionality

### Content Processing Library

#### posts.ts Utilities
```typescript
// Core functions for post processing
getAllPosts(): BlogPost[]
getPostBySlug(slug: string): BlogPost | null
getPostMetadata(): PostMetadata[]
calculateReadTime(content: string): number
sortPostsByDate(posts: BlogPost[]): BlogPost[]
```

#### search.ts Utilities
```typescript
// Search and filtering functions
searchPosts(posts: BlogPost[], query: string): BlogPost[]
filterPostsByTags(posts: BlogPost[], tags: string[]): BlogPost[]
highlightSearchTerms(text: string, query: string): string
```

## Data Models

### Front Matter Schema
```yaml
---
title: "Post Title"
date: "2024-01-15"
author: "Author Name"
excerpt: "Brief description of the post content"
tags: ["technology", "nextjs", "react"]
---
```

### File Naming Convention
- **Format**: `YYYY-MM-DD-post-title.md`
- **Slug Generation**: Derived from filename, removing date prefix
- **URL Structure**: `/posts/post-title`

### Content Organization
- **Posts Directory**: `src/content/posts/`
- **Asset Handling**: Images in `public/posts/` with relative references
- **Category Structure**: Flat structure with tag-based organization

## Error Handling

### Build-time Error Handling
- **Missing Front Matter**: Graceful degradation with default values
- **Invalid Dates**: Fallback to file modification date
- **Malformed Markdown**: Error logging with build failure prevention
- **Missing Files**: 404 page generation for invalid post slugs

### Runtime Error Handling
- **Search Errors**: Fallback to showing all posts
- **Rendering Errors**: Error boundaries for component failures
- **Navigation Errors**: Proper 404 handling for invalid routes

### Validation Strategy
```typescript
// Front matter validation
const validatePostMetadata = (metadata: any): PostMetadata => {
  return {
    title: metadata.title || 'Untitled Post',
    date: metadata.date || new Date().toISOString(),
    author: metadata.author || 'Anonymous',
    excerpt: metadata.excerpt || '',
    tags: Array.isArray(metadata.tags) ? metadata.tags : []
  };
};
```

## Testing Strategy

### Unit Testing Focus
- **Post Processing**: Test markdown parsing and front matter extraction
- **Search Functionality**: Test filtering and search algorithms
- **Component Rendering**: Test component props and state management
- **Utility Functions**: Test date formatting, read time calculation

### Integration Testing
- **Static Generation**: Test build process with sample content
- **Routing**: Test navigation between pages
- **SEO Metadata**: Test meta tag generation
- **Responsive Design**: Test layout across device sizes

### Content Validation Testing
- **Front Matter Parsing**: Test various front matter configurations
- **Markdown Rendering**: Test complex markdown features
- **Error Scenarios**: Test handling of malformed content

### Performance Testing
- **Build Time**: Monitor static generation performance
- **Bundle Size**: Track JavaScript bundle optimization
- **Search Performance**: Test search with large content sets
- **Image Loading**: Test static asset optimization

## SEO and Performance Considerations

### Static Generation Benefits
- **Pre-rendered HTML**: Optimal SEO and loading performance
- **CDN Friendly**: Static files easily cached and distributed
- **Zero Server Requirements**: Deployable to any static hosting

### Metadata Strategy
- **Dynamic Titles**: Post-specific page titles
- **Open Graph**: Social media sharing optimization
- **Structured Data**: JSON-LD for search engine understanding
- **Sitemap Generation**: Automatic sitemap creation during build

### Performance Optimizations
- **Code Splitting**: Automatic splitting by Next.js
- **Image Optimization**: Disabled for static export compatibility
- **CSS Optimization**: Tailwind CSS purging for minimal bundle size
- **Search Optimization**: Client-side search to avoid server requests

## Design Rationale

### Technology Choices
- **Next.js App Router**: Modern routing with better performance and developer experience
- **Static Export**: Eliminates server requirements and reduces hosting costs
- **Tailwind CSS**: Rapid development with consistent design system
- **TypeScript**: Type safety for better maintainability

### Architecture Decisions
- **File-based Content**: Simple content management without database complexity
- **Build-time Processing**: Better performance than runtime markdown processing
- **Client-side Search**: Avoids server requirements while maintaining functionality
- **Component-based Design**: Reusable components for consistent UI

### Scalability Considerations
- **Tag-based Organization**: Flexible categorization without rigid hierarchy
- **Modular Components**: Easy to extend with new features
- **Utility Libraries**: Centralized logic for easy maintenance
- **Static Generation**: Scales well with content growth