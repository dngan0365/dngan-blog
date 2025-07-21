import Link from 'next/link';
import { format } from 'date-fns';
import { BlogPost } from '@/lib/types';

interface PostCardProps {
  post: BlogPost;
}

export default function PostCard({ post }: PostCardProps) {
  return (
    <article className="bg-white rounded-lg shadow-md hover:shadow-lg transition-shadow duration-300 overflow-hidden">
      <div className="p-6">
        <div className="flex items-center justify-between text-sm text-gray-500 mb-2">
          <time dateTime={post.date}>
            {format(new Date(post.date), 'MMMM d, yyyy')}
          </time>
          <span>{post.readTime} min read</span>
        </div>
        
        <h2 className="text-xl font-bold text-gray-900 mb-3 hover:text-blue-600 transition-colors">
          <Link href={`/posts/${post.slug}`}>
            {post.title}
          </Link>
        </h2>
        
        <p className="text-gray-600 mb-4 line-clamp-3">
          {post.excerpt}
        </p>
        
        <div className="mb-4">
          <span className="inline-block bg-purple-100 text-purple-800 text-xs px-2 py-1 rounded-full font-medium">
            {post.category}
          </span>
        </div>
        
        <div className="flex items-center justify-between">
          <div className="flex flex-wrap gap-2">
            {post.tags.slice(0, 3).map((tag) => (
              <span
                key={tag}
                className="inline-block bg-blue-100 text-blue-800 text-xs px-2 py-1 rounded-full"
              >
                {tag}
              </span>
            ))}
            {post.tags.length > 3 && (
              <span className="text-xs text-gray-500">
                +{post.tags.length - 3} more
              </span>
            )}
          </div>
          
          <Link
            href={`/posts/${post.slug}`}
            className="text-blue-600 hover:text-blue-800 font-medium text-sm transition-colors"
          >
            Read more →
          </Link>
        </div>
        
        <div className="mt-4 pt-4 border-t border-gray-100">
          <span className="text-sm text-gray-500">By {post.author}</span>
        </div>
      </div>
    </article>
  );
}