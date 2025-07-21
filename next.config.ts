import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: 'export',
  trailingSlash: true,
  images: {
    unoptimized: true
  },
  // Add base path for GitHub Pages if needed
  // basePath: '/your-repo-name', // Uncomment and update if your repo isn't at root domain
  // assetPrefix: '/your-repo-name/', // Uncomment and update if your repo isn't at root domain
};

export default nextConfig;
