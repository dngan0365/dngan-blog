# AI/ML Tech Blog

A modern, categorized blog focused on Artificial Intelligence and Machine Learning topics, built with Next.js 15, TypeScript, and Tailwind CSS.

## 🚀 Live Demo

Visit the live blog: [Your GitHub Pages URL will be here]

## 📚 Blog Categories

- **🤖 Large Language Models** - GPT, Claude, and modern LLM techniques
- **🗄️ Vector Databases** - Embeddings, similarity search, and vector storage
- **🔗 RAG Systems** - Retrieval-Augmented Generation implementations
- **🧠 Machine Learning** - Fundamental ML algorithms and practices
- **👁️ Computer Vision** - CNNs, Vision Transformers, and image processing
- **💬 Natural Language Processing** - Text processing and language understanding
- **🎨 Web Development** - Modern web technologies and frameworks

## ✨ Features

- **📱 Responsive Design** - Works perfectly on all devices
- **🔍 Advanced Search** - Search posts by content, title, and author
- **🏷️ Tag Filtering** - Filter posts by technical tags
- **📂 Category System** - Browse posts by AI/ML topics
- **⚡ Static Generation** - Fast loading with Next.js static export
- **🎯 SEO Optimized** - Meta tags, Open Graph, and structured data
- **📖 Markdown Support** - Full markdown with syntax highlighting

## 🛠️ Tech Stack

- **Framework**: Next.js 15.4.2 with App Router
- **Language**: TypeScript
- **Styling**: Tailwind CSS 4
- **Content**: Markdown with gray-matter
- **Processing**: remark and remark-html
- **Deployment**: GitHub Pages
- **CI/CD**: GitHub Actions

## 🚀 Getting Started

### Prerequisites
- Node.js 18+ 
- npm or yarn

### Installation

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
```

2. Install dependencies:
```bash
npm install
```

3. Run the development server:
```bash
npm run dev
```

4. Open [http://localhost:3000](http://localhost:3000) in your browser.

## 📝 Adding New Posts

Create a new markdown file in `src/content/posts/` with the following format:

```markdown
---
title: "Your Post Title"
date: "2024-01-XX"
author: "Your Name"
excerpt: "Brief description of the post"
tags: ["tag1", "tag2", "tag3"]
category: "Your Category"
---

# Your Post Title

Your content here...
```

### Available Categories:
- Large Language Models
- Vector Databases  
- RAG Systems
- Machine Learning
- Computer Vision
- Natural Language Processing
- Web Development

## 🏗️ Build and Deploy

### Local Build
```bash
npm run build
```

### Deploy to GitHub Pages
The blog automatically deploys to GitHub Pages when you push to the main branch using GitHub Actions.

## 📁 Project Structure

```
src/
├── app/                 # Next.js App Router
│   ├── layout.tsx       # Root layout
│   ├── page.tsx         # Homepage
│   ├── about/           # About page
│   └── posts/[slug]/    # Dynamic post pages
├── components/          # React components
│   ├── BlogHomepage.tsx # Main homepage component
│   ├── PostCard.tsx     # Post preview cards
│   ├── PostContent.tsx  # Full post display
│   ├── SearchBar.tsx    # Search functionality
│   ├── TagFilter.tsx    # Tag filtering
│   ├── CategoryFilter.tsx # Category filtering
│   └── Header.tsx       # Site header
├── content/posts/       # Markdown blog posts
├── lib/                 # Utility functions
│   ├── posts.ts         # Post processing
│   ├── search.ts        # Search and filtering
│   └── types.ts         # TypeScript types
└── styles/              # Global styles
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/new-post`
3. Add your post in `src/content/posts/`
4. Commit your changes: `git commit -m 'Add new post about X'`
5. Push to the branch: `git push origin feature/new-post`
6. Open a Pull Request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- Built with [Next.js](https://nextjs.org/)
- Styled with [Tailwind CSS](https://tailwindcss.com/)
- Deployed on [GitHub Pages](https://pages.github.com/)
- Markdown processing with [remark](https://remark.js.org/)