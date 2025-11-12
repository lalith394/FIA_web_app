"use client"

import {
  Play,
  SquareMIcon,
  FlaskConical,
  ChevronRight,
} from "lucide-react"
import React from "react"
import {
  SidebarContent,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarMenuSub,
  SidebarMenuSubItem,
  SidebarMenuSubButton,
} from "@/components/ui/sidebar"

const items = [
  {
    title: "Introduction",
    url: "#",
    icon: Play,
    subItems: [
      { title: "About DL models", url: "#" },
      { title: "About Fundus Images", url: "#" },
      { title: "FIA tasks", url: "#" },
    ],
  },
  {
    title: "Methodology",
    url: "#",
    icon: SquareMIcon,
    subItems: [
      { title: "Compact models", url: "#" },
      { title: "Deep Supervision", url: "#" },
      { title: "Autoencoders", url: "#" },
      { title: "Image Classification", url: "#"}
    ],
  },
  {
    title: "Results",
    url: "#",
    icon: FlaskConical,
    subItems: [
      { title: "PCA model comparisons", url: "#" },
      { title: "Deep Supervision", url: "#" },
      { title: "Autoencoders", url: "#" },
    ],
  },
  
]

export function AppSidebar() {
  return (
    <aside
      className="
        group
        fixed left-0 top-0 h-screen
        bg-background border-r
        transition-all duration-300 ease-in-out
        w-16 hover:w-64
        flex flex-col
        overflow-hidden
        z-40
      "
    >
      <SidebarContent className="flex-1 overflow-y-auto">
        <SidebarGroup>
          <SidebarGroupLabel
            className="
              text-sm font-semibold text-muted-foreground
              opacity-0 group-hover:opacity-100 transition-opacity
              ml-2 mt-3
            "
          >
            Documentation
          </SidebarGroupLabel>

          <SidebarGroupContent>
            <SidebarMenu>
              {items.map((item) => (
                <SidebarMenuItem key={item.title}>
                  <SidebarMenuButton
                    asChild
                    className="
                      flex items-center gap-3 px-3 py-2 rounded-lg
                      hover:bg-accent hover:text-primary
                      transition-all w-full justify-between
                    "
                  >
                    <a href={item.url} className="flex items-center w-full">
                      <item.icon className="w-5 h-5 shrink-0" />
                      {/* Label hidden when collapsed */}
                      <span
                        className="
                          ml-3 whitespace-nowrap
                          opacity-0 group-hover:opacity-100
                          hidden group-hover:inline transition-opacity
                          flex-1
                        "
                      >
                        {item.title}
                      </span>
                      {item.subItems && (
                        <ChevronRight className="w-4 h-4 opacity-0 group-hover:opacity-100 transition-opacity" />
                      )}
                    </a>
                  </SidebarMenuButton>

                  {/* Submenu section - only visible when expanded */}
                  {item.subItems && (
                    <SidebarMenuSub
                      className="
                        pl-8 border-l 
                      border-neutral-300 dark:border-neutral-700
                      hidden group-hover:block
                      transition-all duration-200
                      "
                    >
                      {item.subItems.map((sub) => (
                        <SidebarMenuSubItem key={sub.title}>
                          <SidebarMenuSubButton
                            asChild
                            className="
                              text-sm text-muted-foreground hover:text-primary
                            "
                          >
                            <a href={sub.url}>{sub.title}</a>
                          </SidebarMenuSubButton>
                        </SidebarMenuSubItem>
                      ))}
                    </SidebarMenuSub>
                  )}
                </SidebarMenuItem>
              ))}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarContent>
    </aside>
  )
}
