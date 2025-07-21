import BlogHomepage from '@/components/BlogHomepage';
import { getAllPosts } from '@/lib/posts';

export default async function Home() {
  const posts = await getAllPosts();

  return <BlogHomepage initialPosts={posts} />;
}
