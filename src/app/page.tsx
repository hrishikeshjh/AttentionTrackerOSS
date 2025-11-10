'use client';

import { useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { useAuth } from '@/hooks/use-auth';
import AuthForm from '@/components/auth-form';
import { BrainCircuit, Loader2 } from 'lucide-react';

export default function Home() {
  const { user, loading } = useAuth();
  const router = useRouter();

  useEffect(() => {
    if (!loading && user) {
      router.push('/dashboard');
    }
  }, [user, loading, router]);

  if (loading || user) {
    return (
      <div className="flex min-h-screen flex-col items-center justify-center bg-background p-4">
        <Loader2 className="h-16 w-16 animate-spin text-primary" />
        <p className="mt-4 text-muted-foreground">Redirecting...</p>
      </div>
    );
  }

  return (
    <main className="flex min-h-screen flex-col items-center justify-center bg-background p-4">
      <div className="flex flex-col items-center text-center mb-8">
        <BrainCircuit className="h-16 w-16 text-primary mb-4" />
        <h1 className="text-4xl font-bold tracking-tight text-foreground sm:text-5xl font-headline">
          CognEn
        </h1>
        <p className="mt-2 text-lg text-muted-foreground">
          Cognitive Engagement Analyzer
        </p>
      </div>
      <AuthForm />
    </main>
  );
}
