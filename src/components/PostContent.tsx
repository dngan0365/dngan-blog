import Link from 'next/link';
import { format } from 'date-fns';
import { BlogPost } from '@/lib/types';

interface PostContentProps {
  post: BlogPost;
}

export default function PostContent({ post }: PostContentProps) {
  return (
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-4xl mx-auto px-4 py-8">
        <nav className="mb-8">
          <Link
            href="/"
            className="inline-flex items-center text-blue-600 hover:text-blue-800 transition-colors"
          >
            <svg
              className="w-4 h-4 mr-2"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M15 19l-7-7 7-7"
              />
            </svg>
            Back to Blog
          </Link>
        </nav>

        <article className="bg-white rounded-lg shadow-lg overflow-hidden">
          <header className="px-8 py-6 border-b border-gray-200">
            <div className="flex items-center justify-between text-sm text-gray-500 mb-4">
              <time dateTime={post.date}>
                {format(new Date(post.date), 'MMMM d, yyyy')}
              </time>
              <span>{post.readTime} min read</span>
            </div>
            
            <h1 className="text-3xl md:text-4xl font-bold text-gray-900 mb-4">
              {post.title}
            </h1>
            
            <div className="flex items-center justify-between mb-4">
              <p className="text-gray-600">By {post.author}</p>
              
              <span className="inline-block bg-purple-100 text-purple-800 text-sm px-3 py-1 rounded-full font-medium">
                {post.category}
              </span>
            </div>
            
            <div className="flex flex-wrap gap-2">
              {post.tags.map((tag) => (
                <span
                  key={tag}
                  className="inline-block bg-blue-100 text-blue-800 text-sm px-3 py-1 rounded-full"
                >
                  {tag}
                </span>
              ))}
            </div>
          </header>

          <div className="px-8 py-6">
            <div 
              className="prose prose-lg max-w-none prose-headings:text-gray-900 prose-p:text-gray-700 prose-a:text-blue-600 prose-a:no-underline hover:prose-a:underline prose-strong:text-gray-900 prose-code:text-pink-600 prose-code:bg-gray-100 prose-code:px-1 prose-code:py-0.5 prose-code:rounded prose-pre:bg-gray-900 prose-pre:text-gray-100"
              dangerouslySetInnerHTML={{ __html: post.content }}
            />
          </div>
        </article>

        <div className="mt-8 text-center">
          <Link
            href="/"
            className="inline-flex items-center px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
          >
            <svg
              className="w-4 h-4 mr-2"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M15 19l-7-7 7-7"
              />
            </svg>
            Read More Posts
          </Link>
        </div>
      </div>
    </div>
  );
}