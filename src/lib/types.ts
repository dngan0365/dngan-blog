export interface PostMetadata {
  title: string;
  date: string;
  author: string;
  excerpt?: string;
  tags?: string[];
}

export interface BlogPost {
  slug: string;
  title: string;
  date: string;
  author: string;
  excerpt: string;
  tags: string[];
  content: string;
  readTime: number;
}