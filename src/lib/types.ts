export interface PostMetadata {
  title: string;
  date: string;
  author: string;
  excerpt?: string;
  tags?: string[];
  category?: string;
}

export interface BlogPost {
  slug: string;
  title: string;
  date: string;
  author: string;
  excerpt: string;
  tags: string[];
  category: string;
  content: string;
  readTime: number;
}