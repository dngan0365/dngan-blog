import { Metadata } from 'next/types';

export const metadata: Metadata = {
  title: 'About',
  description: 'Learn more about me and this blog.',
};

export default function AboutPage() {
  return (
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-4xl mx-auto px-4 py-8">
        <div className="bg-white rounded-lg shadow-lg overflow-hidden">
          <div className="px-8 py-6">
            <h1 className="text-3xl md:text-4xl font-bold text-gray-900 mb-6">
              About Me
            </h1>
            
            <div className="prose prose-lg max-w-none prose-headings:text-gray-900 prose-p:text-gray-700">
              <p>
                Welcome to my blog! I&apos;m a passionate web developer who loves sharing knowledge 
                and insights about the ever-evolving world of technology.
              </p>
              
              <h2>What You&apos;ll Find Here</h2>
              <p>
                This blog covers a wide range of topics including:
              </p>
              <ul>
                <li>Web development tutorials and best practices</li>
                <li>JavaScript, TypeScript, and modern frameworks</li>
                <li>CSS and design techniques</li>
                <li>Development tools and workflows</li>
                <li>Industry trends and insights</li>
              </ul>
              
              <h2>My Background</h2>
              <p>
                I&apos;ve been working in web development for several years, with experience in 
                both frontend and backend technologies. I believe in writing clean, maintainable 
                code and staying up-to-date with the latest industry standards.
              </p>
              
              <h2>Get in Touch</h2>
              <p>
                I love connecting with fellow developers and readers. Feel free to reach out 
                if you have questions, suggestions, or just want to chat about technology!
              </p>
              
              <p>
                You can find me on various platforms or leave comments on my blog posts. 
                I try to respond to all messages and appreciate any feedback you might have.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}