import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "HRM - 人体 3D 重建与动作驱动",
  description: "多图生成可动画 3D 人体，视频流动作捕获驱动",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="zh-CN">
      <body>{children}</body>
    </html>
  );
}
