import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  reactCompiler: true,
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: 'http://localhost:8000/api/:path*',
      },
      {
        source: '/accounts/:path*',
        destination: 'http://localhost:8000/accounts/:path*',
      },
    ];
  },
};

export default nextConfig;
