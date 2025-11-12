import { SidebarProvider } from "@/components/ui/sidebar"
import { AppSidebar } from "@/app/components/app-sidebar"

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    
      <main className="flex bg-background text-foreground">
        {/* SidebarProvider wraps the sidebar and main */}
        <SidebarProvider>
          <AppSidebar />
          <main className="flex-1 ml-16 transition-all duration-300 ease-in-out group-hover:ml-56">
            {children}
          </main>
        </SidebarProvider>
      </main>
    
  )
}
