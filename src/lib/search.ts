import { BlogPost } from './types';

export function searchPosts(posts: BlogPost[], query: string): BlogPost[] {
  if (!query.trim()) {
    return posts;
  }

  const searchTerm = query.toLowerCase().trim();
  
  return posts.filter(post => {
    const searchableContent = [
      post.title,
      post.excerpt,
      post.content,
      post.author,
      ...post.tags
    ].join(' ').toLowerCase();
    
    return searchableContent.includes(searchTerm);
  });
}

export function filterPostsByTags(posts: BlogPost[], selectedTags: string[]): BlogPost[] {
  if (selectedTags.length === 0) {
    return posts;
  }
  
  return posts.filter(post =>
    selectedTags.some(tag => post.tags.includes(tag))
  );
}

export function highlightSearchTerms(text: string, query: string): string {
  if (!query.trim()) {
    return text;
  }
  
  const regex = new RegExp(`(${query.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})`, 'gi');
  return text.replace(regex, '<mark class="bg-yellow-200">$1</mark>');
}

export function getAllTags(posts: BlogPost[]): { tag: string; count: number }[] {
  const tagCounts = new Map<string, number>();
  
  posts.forEach(post => {
    post.tags.forEach(tag => {
      tagCounts.set(tag, (tagCounts.get(tag) || 0) + 1);
    });
  });
  
  return Array.from(tagCounts.entries())
    .map(([tag, count]) => ({ tag, count }))
    .sort((a, b) => b.count - a.count);
}

export function getAllCategories(posts: BlogPost[]): { category: string; count: number }[] {
  const categoryCounts = new Map<string, number>();
  
  posts.forEach(post => {
    const category = post.category || 'General';
    categoryCounts.set(category, (categoryCounts.get(category) || 0) + 1);
  });
  
  return Array.from(categoryCounts.entries())
    .map(([category, count]) => ({ category, count }))
    .sort((a, b) => b.count - a.count);
}

export function filterPostsByCategory(posts: BlogPost[], selectedCategory: string | null): BlogPost[] {
  if (!selectedCategory) {
    return posts;
  }
  
  return posts.filter(post => post.category === selectedCategory);
}