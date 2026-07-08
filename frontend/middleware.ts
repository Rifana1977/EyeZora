import { NextRequest, NextResponse } from "next/server";

/**
 * Next.js Edge Middleware — protects admin and student routes.
 * Reads the JWT from localStorage is impossible in middleware (edge),
 * so we read from the ez_token cookie set by the login pages.
 */
export function middleware(req: NextRequest) {
  const { pathname } = req.nextUrl;

  // Read token from cookie (set by login pages)
  const token = req.cookies.get("ez_token")?.value;
  const role = req.cookies.get("ez_role")?.value;

  // ── Protect /admin/* routes ──────────────────────────────────────────────
  if (pathname.startsWith("/admin") && !pathname.startsWith("/admin/login")) {
    if (!token || role !== "admin") {
      return NextResponse.redirect(new URL("/admin/login", req.url));
    }
  }

  // ── Protect /student/* routes ────────────────────────────────────────────
  // Public student routes that do NOT require authentication:
  const PUBLIC_STUDENT_ROUTES = [
    "/student/login",
    "/student/forgot-password",
    "/student/reset-password",
  ];
  const isPublicStudentRoute = PUBLIC_STUDENT_ROUTES.some((r) =>
    pathname.startsWith(r)
  );

  if (pathname.startsWith("/student") && !isPublicStudentRoute) {
    if (!token || role !== "student") {
      return NextResponse.redirect(new URL("/student/login", req.url));
    }
  }

  // ── Redirect authenticated users away from login pages ──────────────────
  if (pathname === "/admin/login" && token && role === "admin") {
    return NextResponse.redirect(new URL("/admin/dashboard", req.url));
  }

  if (pathname === "/student/login" && token && role === "student") {
    return NextResponse.redirect(new URL("/student/pre-exam", req.url));
  }

  // ── Old dashboard routes — redirect to new structure ────────────────────
  if (pathname.startsWith("/dashboard/admin")) {
    return NextResponse.redirect(new URL("/admin/dashboard", req.url));
  }
  if (pathname.startsWith("/dashboard/student")) {
    return NextResponse.redirect(new URL("/student/login", req.url));
  }

  return NextResponse.next();
}

export const config = {
  matcher: [
    "/admin/:path*",
    "/student/:path*",
    "/dashboard/:path*",
  ],
};
