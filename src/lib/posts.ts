import fs from 'fs';
import path from 'path';
import matter from 'gray-matter';
import { remark } from 'remark';
import html from 'remark-html';
import { BlogPost, PostMetadata } from './types';

const postsDirectory = path.join(process.cwd(), 'src/content/posts');

export function calculateReadTime(content: string): number {
  const wordsPerMinute = 200;
  const words = content.trim().split(/\s+/).length;
  return Math.ceil(words / wordsPerMinute);
}

export function validatePostMetadata(metadata: Record<string, unknown>): PostMetadata {
  return {
    title: (typeof metadata.title === 'string' ? metadata.title : '') || 'Untitled Post',
    date: (typeof metadata.date === 'string' ? metadata.date : '') || new Date().toISOString(),
    author: (typeof metadata.author === 'string' ? metadata.author : '') || 'Anonymous',
    excerpt: (typeof metadata.excerpt === 'string' ? metadata.excerpt : '') || '',
    tags: Array.isArray(metadata.tags) ? metadata.tags.filter((tag): tag is string => typeof tag === 'string') : []
  };
}

export async function getPostBySlug(slug: string): Promise<BlogPost | null> {
  try {
    const files = fs.readdirSync(postsDirectory);
    const fileName = files.find(file => {
      const fileSlug = file.replace(/^\d{4}-\d{2}-\d{2}-/, '').replace(/\.md$/, '');
      return fileSlug === slug;
    });

    if (!fileName) {
      return null;
    }

    const fullPath = path.join(postsDirectory, fileName);
    const fileContents = fs.readFileSync(fullPath, 'utf8');
    const { data, content } = matter(fileContents);
    
    const validatedMetadata = validatePostMetadata(data);
    const processedContent = await remark().use(html).process(content);
    const contentHtml = processedContent.toString();
    
    return {
      slug,
      title: validatedMetadata.title,
      date: validatedMetadata.date,
      author: validatedMetadata.author,
      excerpt: validatedMetadata.excerpt || content.substring(0, 150) + '...',
      tags: validatedMetadata.tags || [],
      content: contentHtml,
      readTime: calculateReadTime(content)
    };
  } catch (error) {
    console.error(`Error reading post ${slug}:`, error);
    return null;
  }
}

export async function getAllPosts(): Promise<BlogPost[]> {
  try {
    const fileNames = fs.readdirSync(postsDirectory);
    const allPostsData = await Promise.all(
      fileNames
        .filter(fileName => fileName.endsWith('.md'))
        .map(async (fileName) => {
          const slug = fileName.replace(/^\d{4}-\d{2}-\d{2}-/, '').replace(/\.md$/, '');
          return await getPostBySlug(slug);
        })
    );

    return allPostsData
      .filter((post): post is BlogPost => post !== null)
      .sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime());
  } catch (error) {
    console.error('Error reading posts directory:', error);
    return [];
  }
}

export function getPostMetadata(): PostMetadata[] {
  try {
    const fileNames = fs.readdirSync(postsDirectory);
    return fileNames
      .filter(fileName => fileName.endsWith('.md'))
      .map((fileName) => {
        const fullPath = path.join(postsDirectory, fileName);
        const fileContents = fs.readFileSync(fullPath, 'utf8');
        const { data } = matter(fileContents);
        return validatePostMetadata(data);
      })
      .sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime());
  } catch (error) {
    console.error('Error reading post metadata:', error);
    return [];
  }
}

export function sortPostsByDate(posts: BlogPost[]): BlogPost[] {
  return posts.sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime());
}