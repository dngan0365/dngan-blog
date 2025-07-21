'use client';

import { useState, useEffect } from 'react';
import PostCard from '@/components/PostCard';
import SearchBar from '@/components/SearchBar';
import TagFilter from '@/components/TagFilter';
import { BlogPost } from '@/lib/types';
import { searchPosts, filterPostsByTags, getAllTags } from '@/lib/search';

interface BlogHomepageProps {
  initialPosts: BlogPost[];
}

export default function BlogHomepage({ initialPosts }: BlogHomepageProps) {
  const [posts] = useState<BlogPost[]>(initialPosts);
  const [filteredPosts, setFilteredPosts] = useState<BlogPost[]>(initialPosts);
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedTags, setSelectedTags] = useState<string[]>([]);

  useEffect(() => {
    let filtered = posts;
    
    if (searchQuery) {
      filtered = searchPosts(filtered, searchQuery);
    }
    
    if (selectedTags.length > 0) {
      filtered = filterPostsByTags(filtered, selectedTags);
    }
    
    setFilteredPosts(filtered);
  }, [posts, searchQuery, selectedTags]);

  const availableTags = getAllTags(posts);

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-6xl mx-auto px-4 py-8">
        <header className="text-center mb-12">
          <h1 className="text-4xl font-bold text-gray-900 mb-4">
            Welcome to My Blog
          </h1>
          <p className="text-xl text-gray-600 max-w-2xl mx-auto">
            Thoughts, tutorials, and insights about web development, technology, and more.
          </p>
        </header>

        <div className="mb-8 space-y-4">
          <SearchBar
            value={searchQuery}
            onChange={setSearchQuery}
            placeholder="Search posts..."
          />
          
          <TagFilter
            availableTags={availableTags}
            selectedTags={selectedTags}
            onTagToggle={(tag) => {
              setSelectedTags(prev =>
                prev.includes(tag)
                  ? prev.filter(t => t !== tag)
                  : [...prev, tag]
              );
            }}
          />
        </div>

        {filteredPosts.length === 0 ? (
          <div className="text-center py-12">
            <p className="text-gray-600 text-lg">
              {posts.length === 0 
                ? "No posts available yet." 
                : "No posts match your search criteria."}
            </p>
            {(searchQuery || selectedTags.length > 0) && (
              <button
                onClick={() => {
                  setSearchQuery('');
                  setSelectedTags([]);
                }}
                className="mt-4 text-blue-600 hover:text-blue-800 underline"
              >
                Clear filters
              </button>
            )}
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {filteredPosts.map((post) => (
              <PostCard key={post.slug} post={post} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}